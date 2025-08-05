% Load CIFAR-10 Dataset
load('cifar-10-data.mat'); 

% Data has the following variables:
% data: 60000x32x32x3 matrix of image data
% labels: 60000x1 vector of labels
% label_names: 10x1 cell array of category names

% Set the random seed using student ID
studentID = 38546299;
rng(studentID);

% Visualizing 4 random images with their labels
figure;
for i = 1:4
    subplot(1, 4, i);
    idx = randi(60000); 
    imagesc(squeeze(data(idx,:,:,:))); 
    title(label_names{labels(idx) + 1});
end
saveas(gcf, 'example_images.png'); 

% Convert the image data to double for processing
data = double(data);

studentID = 38546299;
rng(studentID);

% Randomly select 3 classes using student ID
numClasses = 10; 
selected_classes = randperm(numClasses, 3);

% Extract the images and labels of the selected classes
selected_indices = ismember(labels, selected_classes);
selected_data = data(selected_indices, :, :, :);
selected_labels = labels(selected_indices);

% Flatten the images into 3072 features (32x32x3)
selected_data = reshape(selected_data, [], 32*32*3);

% Split into training and testing (50% training, 50% testing)
numSamples = length(selected_labels);
training_index = randperm(numSamples, round(numSamples/2)); % Random training indices
testing_index = setdiff(1:numSamples, training_index); % Rest for testing

% Create training and testing sets
training_data = selected_data(training_index, :);
testing_data = selected_data(testing_index, :);
training_labels = selected_labels(training_index);
testing_labels = selected_labels(testing_index);

% Implementing KNN classifier with Euclidean (L2) distance
knnL2 = fitcknn(training_data, training_labels, 'NumNeighbors', 5, 'Distance', 'euclidean');
tic;
predictions_knnL2 = predict(knnL2, testing_data);
knnL2_time = toc;
knnL2_accuracy = sum(predictions_knnL2 == testing_labels) / length(testing_labels);

% Implementing KNN classifier with another distance metric (Cosine)
knnCosine = fitcknn(training_data, training_labels, 'NumNeighbors', 5, 'Distance', 'cosine');
tic;
predictions_knnCosine = predict(knnCosine, testing_data);
knnCosine_time = toc;
knnCosine_accuracy = sum(predictions_knnCosine == testing_labels) / length(testing_labels);

% Train a Support Vector Machine (SVM) for comparison
svmModel = fitcecoc(training_data, training_labels); % Default multiclass SVM
tic;
predictions_svm = predict(svmModel, testing_data);
svm_time = toc;
svm_accuracy = sum(predictions_svm == testing_labels) / length(testing_labels);

% Train a Decision Tree classifier for comparison
treeModel = fitctree(training_data, training_labels);
tic;
predictions_tree = predict(treeModel, testing_data);
tree_time = toc;
tree_accuracy = sum(predictions_tree == testing_labels) / length(testing_labels);


% Create confusion matrices
knnL2_confusion = confusionmat(testing_labels, predictions_knnL2);
knnCosine_confusion = confusionmat(testing_labels, predictions_knnCosine);
svm_confusion = confusionmat(testing_labels, predictions_svm);
tree_confusion = confusionmat(testing_labels, predictions_tree);

% Save the results for submission
save('cw1.mat', 'selected_classes', 'training_index', 'knnL2_accuracy', 'knnL2_time', ...
    'knnCosine_accuracy', 'knnCosine_time', 'svm_accuracy', 'svm_time', 'tree_accuracy', 'tree_time', ...
    'knnL2_confusion', 'knnCosine_confusion', 'svm_confusion', 'tree_confusion');



% Display results
disp('KNN L2 Accuracy:');
disp(knnL2_accuracy);
disp('KNN Cosine Accuracy:');
disp(knnCosine_accuracy);
disp('SVM Accuracy:');
disp(svm_accuracy);
disp('Decision Tree Accuracy:');
disp(tree_accuracy);

figure; 


subplot(2, 2, 1); 
chart1 = confusionchart(knnL2_confusion); 
title('KNN L2 Confusion Matrix'); 
chart1.FontName = 'Arial';
chart1.XLabel = 'Predicted Class'; 
chart1.YLabel = 'True Class';      

subplot(2, 2, 2); 
chart2 = confusionchart(knnCosine_confusion); 
title('KNN Cosine Confusion Matrix'); 
chart2.FontName = 'Arial';
chart2.XLabel = 'Predicted Class'; 
chart2.YLabel = 'True Class';   

subplot(2, 2, 3); 
chart3 = confusionchart(svm_confusion); 
title('SVM Confusion Matrix'); 
chart3.FontName = 'Arial';
chart3.XLabel = 'Predicted Class'; 
chart3.YLabel = 'True Class';   

subplot(2, 2, 4); 
chart4 = confusionchart(tree_confusion); 
title('Decision Tree Confusion Matrix'); 
chart4.FontName = 'Arial';
chart4.XLabel = 'Predicted Class'; 
chart4.YLabel = 'True Class';  

saveas(gcf, 'confusion_matrices.png');
