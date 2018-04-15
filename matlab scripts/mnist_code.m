clear all;

% Add caffe/matlab to you Matlab search PATH to use matcaffe
addpath('/home/lod/master-thesis/matlab');

% Change here parameters for visualization
use_gpu = 1; %-- we will be using CPU mode, change to 1 - for GPU mode

expath = '/home/lod/master-thesis/examples/master-thesis/';
model_def_file = 'train-mnistCAE12sym0302.prototxt';
model_file = 'snapshots_iter_20000.caffemodel';
TestFileMAT = '/home/lod/master-thesis/mnist_test.mat';

% Prepare input data
heightW = 28;
widthW = 28;
TE = 10000;

caffein = zeros(heightW, widthW, 1, TE, 'single');
load(TestFileMAT);

for j=1:TE
    for i=1:heightW for k=1:widthW caffein(i, k, 1, j) = test_X(j, k+(i-1)*widthW); end; end
end

% Work with caffe 
matcaffe_init(use_gpu, strcat(expath, model_def_file), strcat(expath, model_file));
pc2 = caffe('forward', {caffein});
pc2 = pc2{1};
caffe('reset');
pc2 = (squeeze(pc2)');

figure;
gscatter(pc2(:,1), pc2(:,2), test_labels), grid on;
title('Model 2, 2 hidden output neurons');
xlabel('1st dimension');
ylabel('2nd dimension');
saveas(gcf, strcat(model_file, '.Fig1', '.png'));
