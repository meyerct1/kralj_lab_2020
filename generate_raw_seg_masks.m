%First run the python script to generate the chopped data extracted from
%avis
%This finds all the images and segments them placing them in a folder
%'raw_seg' split into resistant and susceptible.

%Note this assumes resistant and susceptible are spelled out in the folder
%names!!!!! 
fils = dir('All*/ready_data/*/*.png');
fils = fils(randperm(length(fils)));
%Make all the directories
mkdir('raw_seg')
mkdir('raw_seg/resistant')
mkdir('raw_seg/susceptible')

%Set FogBank segmentation parameters.  BAsed on tests 05/07/2020. 
fill_holes_bool_oper = 'AND';
hole_max_perct_intensity = 100;
hole_min_perct_intensity = 0;
manual_finetune = 0;
max_hole_size = Inf;
min_cell_size = 50;
min_hole_size = 50;

%Segment in parallel.  Uses ~800MB/pool.
p = gcp('nocreate');
if isempty(p)
    parpool(16);
end

%Begin parallel segmentation
parfor count = 1:length(fils)
    if contains(fils(count).folder,'resistant')
        sub_fol = 'resistant';
    else
        sub_fol = 'susceptible';
    end
    tmp =  split(fils(count).folder,'/');
    fname = strcat(string(tmp(end)),'_',fils(count).name,'_seg.png');
    %If it has been segemented skip
    if exist(strcat('raw_seg/',sub_fol,filesep,fname),'file')~=2
        %Read in the image making it an 8-bit grayscale
        I = rgb2gray(imread([fils(count).folder filesep fils(count).name]));
        try 
            S = EGT_Segmentation(I, min_cell_size, min_hole_size, max_hole_size, hole_min_perct_intensity, hole_max_perct_intensity, fill_holes_bool_oper, manual_finetune);
        catch ME  %Errors occur when the image is blank
            disp(['Error on image ' num2str(count)])
            S = zeros(size(I))
        end
        %Write the image out
        imwrite(S,strcat('raw_seg/',sub_fol,filesep,fname));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%
%Run the Difference imager between frames and save into test and training
%datasets.
%%%%%%%%%%%%%%%%%%%%%%%%
%For all movies
frames = dir('All*/ready_data/*.avi');
frames = frames(randperm(length(frames)));

little_delta = 5; %What is the delta considered.
total_time = 30;
cnt_train = 0; %Counter for train images
cnt_test =  0; %Counter for test  images
rng(123) %For reproducibility set the random seed

mkdir('unet_data')
mkdir(['unet_data/test_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time)])
mkdir(['unet_data/test_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time) '/image'])
mkdir(['unet_data/test_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time) '/label'])
mkdir(['unet_data/test_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time) '/predict'])

mkdir(['unet_data/train_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time)])
mkdir(['unet_data/train_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time) '/image'])
mkdir(['unet_data/train_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time) '/label'])

for fr=1:length(frames)
    disp(fr)
    %Find movie and frame names
    frm = frames(fr).name;
    mov = frames(fr).folder;
    if contains(frames(fr).folder,'resistant')
        sub_fol = 'resistant';
    else
        sub_fol = 'susceptible';
    end
    %For the first 30 min take the difference between consecuative frames.
    for i=0:total_time-little_delta-1
        f1 = dir([mov filesep frm filesep 'frame' num2str(i) '.*']);
        f2 = dir([mov filesep frm filesep 'frame' num2str(i+little_delta) '.*']);
        for j=1:length(f1)
            im1 = imread([f1(j).folder filesep f1(j).name]);
            im2 = imread([f2(j).folder filesep f2(j).name]);
            diff = rgb2gray(uint8(abs(double(im1)-double(im2))));
            seg1 = imread(['raw_seg/' sub_fol filesep frm '_' f1(j).name '_seg.png']);
            seg2 = imread(['raw_seg/' sub_fol filesep frm '_' f2(j).name '_seg.png']);
            seg = seg1+seg2;
            seg = seg>0;
            %seg = seg.*127;
            if strcmp(sub_fol,'resistant')
                seg = seg.*2;
            end
            %Split the data 80/20
            if rand()<0.8
                sub_sub_fol = ['train_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time) filesep];
                imwrite(uint8(seg),strcat('unet_data/',sub_sub_fol,'label/',num2str(cnt_train),'.png'));
                imwrite(uint8(diff),strcat('unet_data/',sub_sub_fol,'image/',num2str(cnt_train),'.png'));
                cnt_train = cnt_train + 1;
            else
                sub_sub_fol = ['test_little-delta_' num2str(little_delta) '_total-time_' num2str(total_time) filesep];
                imwrite(uint8(seg),strcat('unet_data/',sub_sub_fol,'label/',num2str(cnt_test),'.png'));
                imwrite(uint8(diff),strcat('unet_data/',sub_sub_fol,'image/',num2str(cnt_test),'.png'));
                cnt_test = cnt_test + 1;
            end
        end
    end
end

            
            
            
            
            
    
    
