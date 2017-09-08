import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imsave, imread
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import shutil
from PIL import Image

import torch
from torch.autograd import Variable


def get_predict(model,image,box_side):
    im = image
    
    image = np.transpose(image,(2,0,1))
    image = np.reshape(image,(1,)+image.shape)
    
    mask = np.zeros((image.shape[2],image.shape[3]))

    (x,y) = np.nonzero(np.sum(im!=15,axis=2))
    top_left_x, top_left_y = x[0],y[0]
    bottom_right_x, bottom_right_y = x[-1],y[-1]

    a=top_left_x
    while a+box_side < bottom_right_x: 
        b=top_left_y
        while b+box_side < bottom_right_y:
            sub_image = image[:,:,a:a+box_side,b:b+box_side]
            sub_image = Variable(torch.cuda.FloatTensor(sub_image/255.0),volatile=True) 
            
#             print(model(sub_image) - model2(sub_image))

            output = model(sub_image).data.max(1)[1].cpu().numpy()[0][0] 
#             print(output)
            mask[a:a+box_side,b:b+box_side] = output

            b += box_side

        a += box_side
        
    seg_img = np.multiply(im,np.transpose(np.tile(mask,(3,1,1)),(1,2,0))).astype('uint8')
    
    return (mask,seg_img)



def IOU(p_mask, g_mask):    
    intersection = np.sum((p_mask+g_mask) == 2.0)      
    union = np.sum((p_mask+g_mask) > 0.0)    
    
    return intersection * 1.0 / union

def evaluate_test_iou(img_folder_path, gt_folder_path, model):    
    box_side = 51    
    iou = np.array([])
     
    for (i,filename) in enumerate(os.listdir(img_folder_path)):
        if filename.endswith(".png"):
            image = np.asarray(Image.open(os.path.join(img_folder_path, filename)))
            gt_mask = np.asarray(Image.open(os.path.join(gt_folder_path, filename)))
            gt_mask = (gt_mask>0).astype('uint8')
            
            seg_mask,seg_img = get_predict(model,image,box_side)
            
            iou = np.concatenate((iou, np.array([IOU(seg_mask,gt_mask)])))
                        
            if i%50 == 0:
                print('completed {} images',i)
                    
    return iou

def save_checkpoint(state, prev_best, is_best, path, filename='checkpoint.pkl'):
    torch.save(state, path+filename)    
    if is_best:
        if prev_best:
            os.remove(path + 'model_best_' + str(round(prev_best,2))+'.pkl')            
        shutil.copyfile(path+filename, path + 'model_best_' + str(round(state['best_prec1'],2)) + '.pkl')

def adjust_learning_rate(optimizer, lr_init, decay, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init / (1 + decay * lr_init)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate_model_and_append(path,performance,history):          
    try:     
        history['train_loss']	   += [performance['train_loss']]
    	history['test_loss']	   += [performance['test_loss']]
    	history['train_acc']	   += [performance['train_acc']]
    	history['test_acc']	       += [performance['test_acc']]

    except:   
        with open(path + 'log.csv', 'w') as f:
            f.write('train_loss,test_loss,train_acc,test_acc\n')               

        history['train_loss']	   = [performance['train_loss']]
    	history['test_loss']	   = [performance['test_loss']]
    	history['train_acc']	   = [performance['train_acc']]
    	history['test_acc']	       = [performance['test_acc']]

    
    with open(path + 'log.csv', 'a') as f:
        f.write('{0},{1},{2},{3}\n'.format(history['train_loss'][-1],\
		                                   history['test_loss'][-1],\
		                                   history['train_acc'][-1],\
		                                   history['test_acc'][-1]))
    
    return history
            
           

def plot_performance(path, history):
    full_path = path + 'plots/'    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        
    plt.figure()
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Loss v/s Epochs')
    plt.ylabel('M.S.E Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')    
    plt.savefig(full_path+'loss.png', bbox_inches='tight')
        
    plt.figure()
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Accuracy v/s Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train'], loc='upper left')   
    plt.legend(['train', 'test'], loc='upper left')    
    plt.savefig(full_path+'test_acc.png', bbox_inches='tight')    

    
    
    plt.close('all')
