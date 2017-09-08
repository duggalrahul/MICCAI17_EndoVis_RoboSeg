clear all;
close all;
clc;

box_side = 50; %actual size is box_side+1

dirs = dir('./ground_truth/*');
dirs = {dirs.name};

num_images = size(dir(['./ground_truth/',dirs{3},'/*']),1) - 3;

parfor i=0:num_images    
    fprintf('working on image %d\n',i);
    
    image = imread(['./left_frames/',sprintf('frame%03d.png',[i])]);
    mask = uint8(zeros([size(image,1),size(image,2)]));
        
    for j=3:size(dirs,2)    
        part_mask = imread(['./ground_truth/',dirs{j},sprintf('/frame%03d.png',[i])]);
%         part_mask(part_mask==40) = 0; % remove 'other' instruments
        mask = mask + part_mask;
    end 
    
    imwrite(mask,['foreground/',sprintf('frame%03d.png',[i])]);
    
    cnt_fg=1;
    cnt_bg=1;
        
    [x,y] = find(imfill(sum(image>15,3)));
    top_left_x = x(1); top_left_y = y(1);
    bottom_right_x = x(size(x,1)); bottom_right_y = y(size(y,1));
    
    a=top_left_x; 
    while a+box_side <= bottom_right_x
        b=top_left_y;
        while b+box_side <= bottom_right_y            
            sub_image =  image(a:a+box_side,b:b+box_side,:);
            sub_mask  =  mask(a:a+box_side,b:b+box_side,:); 
            
            b = b+box_side;
            
            if sum(sum(sub_mask))
                imwrite(sub_image,sprintf('fg/set_1_frame%03d_%d_%04d.png',i,cnt_fg,sum(sum(sub_mask>0))));
                cnt_fg = cnt_fg+1;
            else
                imwrite(sub_image,sprintf('bg/set_1_frame%03d_%d.png',i,cnt_bg));
                cnt_bg = cnt_bg+1;
            end
        end
        a = a+box_side;
    end   
end

%%%%%% filter bg images %%%%%%%%
bg_fg_ratio = 1/1;

num_fg = size(dir(['./fg/*']),1);
num_bgf = num_fg * bg_fg_ratio;

bg = dir('./bg/*');
bg = {bg.name};

bgf = randsample(bg,num_bgf);

parfor i=1:size(bgf,2)
    if strcmp(bgf{i},'.') || strcmp(bgf{i},'..')
        bgf{i}
        continue;
    end
    
    img = imread(['bg/',bgf{i}]);
    imwrite(img,['bgf/',bgf{i}]);
end



