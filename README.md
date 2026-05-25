# Waste_Classification_Multiclass_CNN_Pytorch

**ABSTRACT**

Improper waste segregation remains a major challenge in modern waste management systems, leading to inefficient recycling, increased environmental pollution and higher operational costs. This project addresses the waste segregation task as a computer vision image classification problem and formulated as a multi-class, single-label supervised learning task, where waste images are categorized into classes such as plastic, metal, glass, paper, food organic waste, and other materials using the RealWaste dataset. To achieve this objective, transfer learning is applied using pre-trained Convolutional Neural Network (CNN) architectures – ResNet34 as a baseline, ResNet18 as the primary model, and EfficientNet-B0 as the final optimized model which have demonstrated strong performance on ImageNet. By leveraging pre-trained weights from large-scale image datasets, these models are fine-tuned on the RealWaste dataset to learn material-specific visual features such as colour, texture and structural patterns to enable accurate classification. Performance is evaluated using standard classification metrics including accuracy, precision, recall and F1-score, with results indicating that transfer learning significantly improves convergence and overall classification performance. Among these evaluated models with RealWaste dataset, ResNet18 and EfficientNet-B0 showed strong overall performance, compared to the baseline ResNet34. The findings confirm that deep transfer learning approaches are highly effective for automated waste segregation and offer a practical solution for intelligent recycling systems.


**MOTIVATION**

This project is motivated by the desire to develop a real-world computer vision application that demonstrates the practical use of CNN-based deep learning concepts. Waste segregation presents a meaningful multi-class image classification problem, going beyond simple binary classification and offering a more realistic and challenging scenario. Improper waste sorting contributes to environmental pollution and reduces recycling efficiency, making it an important sustainability issue. With the availability of the RealWaste dataset, this problem provided an opportunity to apply transfer learning techniques using advanced CNN architectures. Personal interest in leveraging artificial intelligence for environmental sustainability further inspired the development of an automated waste classification system. 


**INTRODUCTION**

Effective recycling systems rely heavily on accurate waste segregation. However, conventional sorting methods depend largely on manual labour, which can be inconsistent, time-consuming, and costly. The integration of computer vision and deep learning offers a scalable alternative, capable of automatically identifying waste materials from images. In this project, waste segregation is addressed as a supervised image classification task using the RealWaste dataset. Deep Convolutional Neural Networks is applied to learn distinguishing visual characteristics of different waste categories. Instead of training models from scratch, transfer learning is utilized to leverage knowledge from large-scale datasets, enabling faster convergence and improved generalization. By adapting established architectures such as ResNet18, ResNet34, and EfficientNet-B0, the proposed system aims to provide a reliable and practical solution for intelligent waste management applications.


**DATA COLLECTION AND PREPROCESSING**

In supervised deep learning tasks, data collection and pre-processing form the foundation of model performance. Since supervised learning relies on labelled data, the quality, diversity, and correct annotation of the dataset directly influence the model’s ability to generalize. Therefore, the first step is to choose the right dataset of the expected quality. 

For this project, the RealWaste dataset is chosen. It is a publicly available image dataset designed specifically for waste classification tasks. RealWaste is particularly well-suited for this task for several reasons. First, it reflects real-world waste scenarios, which improves the practical relevance of the trained model. Second, it supports a multi-class classification setup, aligning well with the objective of building a robust waste segregation system rather than a simple binary classifier. Third, the dataset size and class diversity make it appropriate for transfer learning using pre-trained CNN architectures such as ResNet, EfficientNet etc. Finally, the availability of labelled data ensures compatibility with supervised learning approaches, enabling effective fine-tuning and reliable performance evaluation.

It is an image classification dataset with colour images of waste items across 9 major material types, collected within an authentic landfill environment. These images are released in 524x524 resolution. Under the proposed labels, image counts are as shown in Fig. 1 along with the other basic details. 

<img width="626" height="308" alt="image" src="https://github.com/user-attachments/assets/cc7f304a-bff4-4937-a2f3-7525936f5a07" />

Unlike synthetic or highly controlled datasets, RealWaste images are captured in realistic environments with variations in lighting conditions, background clutter, object orientation, and scale.

Fig. 2 shows the class frequency distribution of the images.

<img width="875" height="291" alt="image" src="https://github.com/user-attachments/assets/7eb907d1-39e7-4ded-98c2-3945dfa599dd" />

The full dataset is then split into training (70%), validation (15%) and test (15%) sets before the pre-processing is applied. The splitting is done using a fixed seed of 42, which ensures that the split is always the same every time the code is run. The pre-processing typically includes data cleaning, resizing images to a fixed input dimension (to 224x224 for transfer learning based on ImageNet), normalization of pixel values (using the mean & standard deviation of ImageNet dataset) and data augmentation (such as rotation, flipping etc). Particularly the augmentation is done only for the training set which increases the training dataset by generating more but similar images, which is necessary if oversampling is to be applied to an unbalanced dataset. 

The following specific augmentations are performed using TorchVision transforms:

          o	RandomResizedCrop: Improves scale and position invariance.
          
          o	RandomHorizontalFlip: Makes model robust to left–right orientation changes.
          
          o	RandomRotation: Adds rotation invariance.
          
          o	ColorJitter: Improves robustness to lighting and colour variations.
          
          o	Lastly transformed to Tensor datatype for further processing.


<img width="830" height="446" alt="image" src="https://github.com/user-attachments/assets/90d9ec16-23f8-4ca1-a2af-a6295cb79a3e" />

These steps help improve convergence, reduce overfitting, and ensure that the model learns meaningful patterns rather than noise. Images need to be presented in batches during the training, as the whole dataset at once is not computationally feasible. Images can’t just be presented in their order, they need to be shuffled and presented in “mixed” batches.


**MODEL SELECTION AND TRAINING INITIALIZATION**

Based on recommendations for image classification, below three pretrained models were initially selected for this task of waste image classification:

<img width="887" height="436" alt="image" src="https://github.com/user-attachments/assets/aa17de33-a76b-4a05-ad6a-bcdeffc809f0" />

Hyperparameter tuning also were indicative about the choice of ResNet18 as baseline and ResNet34 as the primary model. 

**Note**: Although ResNet18 was initially selected as the baseline due to its simpler architecture, experimental results on the RealWaste dataset (which is limited) showed that it outperformed ResNet34 across all evaluation metrics. Therefore, **ResNet18 is treated as the primary model** here, while **ResNet34 serves as the baseline for comparison**, highlighting the effectiveness of shallower architectures for this task.

Due to the limited size of the dataset, a **transfer learning** approach was adopted using a pre-trained ResNet18 (also ResNet34 and EfficientNet-B0) model. Transfer learning enables leveraging rich feature representations learned from large-scale datasets, resulting in stronger performance and faster convergence. The convolutional base of ResNet18 is loaded with pre-trained weights and kept frozen to preserve the learned feature extraction capabilities. The final fully connected layer is replaced with a custom classifier tailored to the target number of classes (9 classes to be precise). Only this newly added classifier layer is trained, while the earlier layers remained unchanged. This approach reduces computational cost, accelerates training, and improves generalization, particularly when working with limited training data.

The optimizer is configured to update only the trainable parameters by filtering layers with requires_grad=True, ensuring that only the intended (unfrozen) layers are optimized during training.


**MODEL TRAINING AND VALIDATION EXPERIMENTS**

The model training and validation are performed on a **local machine (with CPU) which took approximately ~1.5 hours**. The training phase is used to optimize the model parameters by minimizing the loss function, while the validation phase evaluates the model’s performance on unseen data during training. This helps monitor generalization ability and detect overfitting.

      o	This process is done one by one for both selected models (ResNet18, ResNet34, and later EfficientNet-B0) for 10 number of epochs using the standard CNN training procedure. Also, each time a different optimizer (from Adam and SGD) is used for further evaluation.
      
      o	Other configuration based on the findings of hyperparameter training is set as below:
      Batch size: 32, learning rate: 0.001 and the CrossEntropyLoss as the loss function.
      
      o	The steps of the training loop start as: for each epoch and for each batch in the training dataset, a forward pass is performed producing an output tensor of shape [batch size, number of classes].
      
      o	The loss is computed between predicted outputs and ground truth labels using the loss function.
      
      o	Backpropagation is carried out to compute gradients of the loss with respect to model parameters.
      
      o	The optimizer updates the model weights based on the computed gradients.
      
      o	For each epoch, the average training loss and average training accuracy are calculated and recorded.
      
      o	After each training epoch, validation is performed using the validation dataset.
      
      o	For each epoch and for each batch in the validation dataset, only a forward pass is executed without gradient computation or weight updates.
      
      o	The loss is computed to evaluate model performance, but no backpropagation is performed.
      
      o	For each epoch, the average validation loss and average validation accuracy are calculated and recorded.
      
      o	At the end of training, the trained model weights are saved as a checkpoint file to enable future evaluation, reproducibility, and potential further fine-tuning or deployment.


**<ins>Tuning Results:</ins>**

<img width="741" height="341" alt="image" src="https://github.com/user-attachments/assets/b5b0ce9c-f4d5-435f-89ee-b1ee6cd411b9" />
<img width="741" height="320" alt="image" src="https://github.com/user-attachments/assets/50ab0edf-a4ab-422a-98ff-ec94d018185f" />
<img width="741" height="319" alt="image" src="https://github.com/user-attachments/assets/21cfa5e0-1c6e-4a5a-8fc8-07923cfbe9d7" />


**<ins>Training Results:</ins>**

**Model: Resnet18, Optimizer: Adam, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Epoch [1/10]
Train Loss: 1.4875 | Train Acc: 0.5105 || Val Loss: 1.0609 | Val Acc: 0.6685

Epoch [2/10]
Train Loss: 0.9183 | Train Acc: 0.7060 || Val Loss: 0.8947 | Val Acc: 0.6896

Epoch [3/10]
Train Loss: 0.7671 | Train Acc: 0.7589 || Val Loss: 0.7970 | Val Acc: 0.7275

Epoch [4/10]
Train Loss: 0.7090 | Train Acc: 0.7733 || Val Loss: 0.7780 | Val Acc: 0.7289

Epoch [5/10]
Train Loss: 0.6497 | Train Acc: 0.7886 || Val Loss: 0.7384 | Val Acc: 0.7486

Epoch [6/10]
Train Loss: 0.6210 | Train Acc: 0.7949 || Val Loss: 0.7083 | Val Acc: 0.7486

Epoch [7/10]
Train Loss: 0.5885 | Train Acc: 0.8073 || Val Loss: 0.6725 | Val Acc: 0.7711

Epoch [8/10]
Train Loss: 0.5578 | Train Acc: 0.8184 || Val Loss: 0.6601 | Val Acc: 0.7711

Epoch [9/10]
Train Loss: 0.5399 | Train Acc: 0.8286 || Val Loss: 0.6684 | Val Acc: 0.7669

Epoch [10/10]
Train Loss: 0.5217 | Train Acc: 0.8307 || Val Loss: 0.6453 | Val Acc: 0.7809



**Model: Resnet34, Optimizer: Adam, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Epoch [1/10]
Train Loss: 1.5161 | Train Acc: 0.4838 || Val Loss: 1.0724 | Val Acc: 0.6559

Epoch [2/10]
Train Loss: 0.9527 | Train Acc: 0.7014 || Val Loss: 0.8885 | Val Acc: 0.7205

Epoch [3/10]
Train Loss: 0.8055 | Train Acc: 0.7408 || Val Loss: 0.7935 | Val Acc: 0.7430

Epoch [4/10]
Train Loss: 0.7289 | Train Acc: 0.7565 || Val Loss: 0.7362 | Val Acc: 0.7542

Epoch [5/10]
Train Loss: 0.6668 | Train Acc: 0.7874 || Val Loss: 0.7131 | Val Acc: 0.7514

Epoch [6/10]
Train Loss: 0.6355 | Train Acc: 0.7916 || Val Loss: 0.6835 | Val Acc: 0.7711

Epoch [7/10]
Train Loss: 0.5966 | Train Acc: 0.8070 || Val Loss: 0.6797 | Val Acc: 0.7626

Epoch [8/10]
Train Loss: 0.5753 | Train Acc: 0.8112 || Val Loss: 0.6353 | Val Acc: 0.7893

Epoch [9/10]
Train Loss: 0.5539 | Train Acc: 0.8154 || Val Loss: 0.6399 | Val Acc: 0.7725

Epoch [10/10]
Train Loss: 0.5388 | Train Acc: 0.8235 || Val Loss: 0.6349 | Val Acc: 0.7739


**Model: Resnet18, Optimizer: SGD, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Epoch [1/10]
Train Loss: 1.6352 | Train Acc: 0.4447 || Val Loss: 1.2033 | Val Acc: 0.6250

Epoch [2/10]
Train Loss: 1.0579 | Train Acc: 0.6795 || Val Loss: 1.0016 | Val Acc: 0.6615

Epoch [3/10]
Train Loss: 0.8972 | Train Acc: 0.7225 || Val Loss: 0.9109 | Val Acc: 0.7079

Epoch [4/10]
Train Loss: 0.8099 | Train Acc: 0.7495 || Val Loss: 0.8346 | Val Acc: 0.7022

Epoch [5/10]
Train Loss: 0.7521 | Train Acc: 0.7619 || Val Loss: 0.8015 | Val Acc: 0.7402

Epoch [6/10]
Train Loss: 0.7069 | Train Acc: 0.7739 || Val Loss: 0.7677 | Val Acc: 0.7458

Epoch [7/10]
Train Loss: 0.6825 | Train Acc: 0.7856 || Val Loss: 0.7444 | Val Acc: 0.7416

Epoch [8/10]
Train Loss: 0.6442 | Train Acc: 0.7946 || Val Loss: 0.7207 | Val Acc: 0.7472

Epoch [9/10]
Train Loss: 0.6276 | Train Acc: 0.7913 || Val Loss: 0.7023 | Val Acc: 0.7598

Epoch [10/10]
Train Loss: 0.6086 | Train Acc: 0.8010 || Val Loss: 0.6934 | Val Acc: 0.7626


**Model: Resnet34, Optimizer: SGD, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Epoch [1/10]
Train Loss: 1.6471 | Train Acc: 0.4465 || Val Loss: 1.1980 | Val Acc: 0.6194

Epoch [2/10]
Train Loss: 1.0852 | Train Acc: 0.6575 || Val Loss: 0.9555 | Val Acc: 0.7065

Epoch [3/10]
Train Loss: 0.9140 | Train Acc: 0.7186 || Val Loss: 0.8672 | Val Acc: 0.7079

Epoch [4/10]
Train Loss: 0.8110 | Train Acc: 0.7474 || Val Loss: 0.8150 | Val Acc: 0.7388

Epoch [5/10]
Train Loss: 0.7553 | Train Acc: 0.7631 || Val Loss: 0.7703 | Val Acc: 0.7458

Epoch [6/10]
Train Loss: 0.7125 | Train Acc: 0.7808 || Val Loss: 0.7471 | Val Acc: 0.7346

Epoch [7/10]
Train Loss: 0.6902 | Train Acc: 0.7715 || Val Loss: 0.7138 | Val Acc: 0.7542

Epoch [8/10]
Train Loss: 0.6637 | Train Acc: 0.7868 || Val Loss: 0.7103 | Val Acc: 0.7486

Epoch [9/10]
Train Loss: 0.6376 | Train Acc: 0.7946 || Val Loss: 0.6933 | Val Acc: 0.7683

Epoch [10/10]
Train Loss: 0.6263 | Train Acc: 0.8001 || Val Loss: 0.6761 | Val Acc: 0.7640


**Model: EfficientNet-B0, Optimizer: SGD, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Epoch [1/10]
Train Loss: 1.2779 | Train Acc: 0.5884 || Val Loss: 0.8364 | Val Acc: 0.7416

Epoch [2/10]
Train Loss: 0.8042 | Train Acc: 0.7444 || Val Loss: 0.7096 | Val Acc: 0.7654

Epoch [3/10]
Train Loss: 0.6783 | Train Acc: 0.7847 || Val Loss: 0.6610 | Val Acc: 0.7781

Epoch [4/10]
Train Loss: 0.6201 | Train Acc: 0.7940 || Val Loss: 0.6338 | Val Acc: 0.7893

Epoch [5/10]
Train Loss: 0.5606 | Train Acc: 0.8199 || Val Loss: 0.6179 | Val Acc: 0.7837

Epoch [6/10]
Train Loss: 0.5618 | Train Acc: 0.8202 || Val Loss: 0.5950 | Val Acc: 0.7879

Epoch [7/10]
Train Loss: 0.5154 | Train Acc: 0.8310 || Val Loss: 0.5694 | Val Acc: 0.8020

Epoch [8/10]
Train Loss: 0.4972 | Train Acc: 0.8331 || Val Loss: 0.5692 | Val Acc: 0.7992

Epoch [9/10]
Train Loss: 0.4882 | Train Acc: 0.8322 || Val Loss: 0.5493 | Val Acc: 0.8118

Epoch [10/10]
Train Loss: 0.4818 | Train Acc: 0.8391 || Val Loss: 0.5536 | Val Acc: 0.8048


**<ins>Evaluation Results:</ins>**

**Model: Resnet18, Optimizer: Adam, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Accuracy: 0.7899
Precision: 0.7906
Recall: 0.7899
F1_score: 0.7897

**Model: Resnet34, Optimizer: Adam, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Accuracy: 0.7507
Precision: 0.7560
Recall: 0.7507
F1_score: 0.7496

**Model: Resnet18, Optimizer: SGD, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Accuracy: 0.7661
Precision: 0.7659
Recall: 0.7661
F1_score: 0.7631

**Model: Resnet34, Optimizer: SGD, Batch Size: 32, Learning Rate: 0.001,  Number of Epochs: 10:**

Accuracy: 0.7241
Precision: 0.7281
Recall: 0.7241
F1_score: 0.7230

**Model: EfficientNet-B0, Optimizer: SGD, Batch Size: 32, Learning Rate: 0.01,  Number of Epochs: 10:**

Accuracy: 0.7899
Precision: 0.7910
Recall: 0.7899
F1_score: 0.7890

