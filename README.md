# PROJECT TITILE

## Deep Learning Project for Waste Segregation Through Multiclass Image Classification using CNN and Pytorch:

# ABSTRACT

Improper waste segregation remains a major challenge in modern waste management systems, leading to inefficient recycling, increased environmental pollution and higher operational costs. This project addresses the waste segregation task as a computer vision image classification problem and formulated as a multi-class, single-label supervised learning task, where waste images are categorized into classes such as plastic, metal, glass, paper, food organic waste, and other materials using the RealWaste dataset. To achieve this objective, transfer learning is applied using pre-trained Convolutional Neural Network (CNN) architectures – ResNet34 as a baseline, ResNet18 as the primary model, and EfficientNet-B0 as the final optimized model which have demonstrated strong performance on ImageNet. By leveraging pre-trained weights from large-scale image datasets, these models are fine-tuned on the RealWaste dataset to learn material-specific visual features such as colour, texture and structural patterns to enable accurate classification. Performance is evaluated using standard classification metrics including accuracy, precision, recall and F1-score, with results indicating that transfer learning significantly improves convergence and overall classification performance. Among these evaluated models with RealWaste dataset, ResNet18 and EfficientNet-B0 showed strong overall performance, compared to the baseline ResNet34. The findings confirm that deep transfer learning approaches are highly effective for automated waste segregation and offer a practical solution for intelligent recycling systems.


# MOTIVATION

This project is motivated by the desire to develop a real-world computer vision application that demonstrates the practical use of CNN-based deep learning concepts. Waste segregation presents a meaningful multi-class image classification problem, going beyond simple binary classification and offering a more realistic and challenging scenario. Improper waste sorting contributes to environmental pollution and reduces recycling efficiency, making it an important sustainability issue. With the availability of the RealWaste dataset, this problem provided an opportunity to apply transfer learning techniques using advanced CNN architectures. Personal interest in leveraging artificial intelligence for environmental sustainability further inspired the development of an automated waste classification system. 


# INTRODUCTION

Effective recycling systems rely heavily on accurate waste segregation. However, conventional sorting methods depend largely on manual labour, which can be inconsistent, time-consuming, and costly. The integration of computer vision and deep learning offers a scalable alternative, capable of automatically identifying waste materials from images. In this project, waste segregation is addressed as a supervised image classification task using the RealWaste dataset. Deep Convolutional Neural Networks is applied to learn distinguishing visual characteristics of different waste categories. Instead of training models from scratch, transfer learning is utilized to leverage knowledge from large-scale datasets, enabling faster convergence and improved generalization. By adapting established architectures such as ResNet18, ResNet34, and EfficientNet-B0, the proposed system aims to provide a reliable and practical solution for intelligent waste management applications.


# DATA COLLECTION AND PREPROCESSING

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


# MODEL SELECTION AND TRAINING INITIALIZATION

Based on recommendations for image classification, below three pretrained models were initially selected for this task of waste image classification:

<img width="887" height="436" alt="image" src="https://github.com/user-attachments/assets/aa17de33-a76b-4a05-ad6a-bcdeffc809f0" />

Hyperparameter tuning also were indicative about the choice of ResNet18 as baseline and ResNet34 as the primary model. 

**Note**: Although ResNet18 was initially selected as the baseline due to its simpler architecture, experimental results on the RealWaste dataset (which is limited) showed that it outperformed ResNet34 across all evaluation metrics. Therefore, **ResNet18 is treated as the primary model** here, while **ResNet34 serves as the baseline for comparison**, highlighting the effectiveness of shallower architectures for this task.

Due to the limited size of the dataset, a **transfer learning** approach was adopted using a pre-trained ResNet18 (also ResNet34 and EfficientNet-B0) model. Transfer learning enables leveraging rich feature representations learned from large-scale datasets, resulting in stronger performance and faster convergence. The convolutional base of ResNet18 is loaded with pre-trained weights and kept frozen to preserve the learned feature extraction capabilities. The final fully connected layer is replaced with a custom classifier tailored to the target number of classes (9 classes to be precise). Only this newly added classifier layer is trained, while the earlier layers remained unchanged. This approach reduces computational cost, accelerates training, and improves generalization, particularly when working with limited training data.

The optimizer is configured to update only the trainable parameters by filtering layers with requires_grad=True, ensuring that only the intended (unfrozen) layers are optimized during training.


# MODEL TRAINING AND VALIDATION EXPERIMENTS

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


**<ins>Detailed Training Results with varied models and settings:</ins>**

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


# EXPERIMENT RESULTS PLOTTING

The below graph (Fig. 4) depicts the train and validation loss across epochs:

          o  Training loss steadily decreases: The model is learning patterns from the training data effectively.
          
          o  Validation loss also decreases initially: The model is generalizing well in the early epochs.
          
          o  Gap between train and validation loss appears after ~epoch 4–5: Indicates slight overfitting is beginning.
          
          o  Validation loss flattens around later epochs: Learning improvements slow down; model reaches near-optimal performance.
          
          o  Small fluctuation in validation loss near the end: Normal behaviour due to dataset variability but signals mild overfitting.


<img width="430" height="352" alt="image" src="https://github.com/user-attachments/assets/39d3dc3c-4035-414e-b033-c541fed607ba" />

Overall, the model is learning well with good generalization, but slight overfitting may start after mid-training. Early stopping around epoch 7–8 might be optimal.

The Accuracy vs Epochs graph (Fig. 5) depicts:

          o  Training accuracy increases steadily: The model is learning patterns effectively from the training data.
          
          o  Validation accuracy also improves over epochs: The model generalizes well to unseen data.
          
          o  Gap between training and validation accuracy appears after ~epoch 4–5: Indicates mild overfitting is starting.
          
          o  Validation accuracy stabilizes around 0.77–0.78: Performance improvement slows; model is reaching its generalization limit.
          
          o  No sharp drop in validation accuracy: No severe overfitting; training remains stable.


<img width="407" height="341" alt="image" src="https://github.com/user-attachments/assets/bc0e544e-fa0b-4ebf-91b2-4a0559527ffe" />

The model shows good learning and reasonable generalization. Slight overfitting begins in later epochs, and the best validation performance appears around epoch 8–10.


# HYPERPARAMETER TUNING

Hyperparameter tuning plays a crucial role in deep learning models, as it directly affects convergence speed, model stability, and overall generalization performance. In the task of waste segregation using convolutional neural networks, appropriate selection of hyperparameters such as learning rate, batch size, number of epochs, and optimizer is essential to achieve reliable classification performance across multiple waste categories. Below are the values which were used to review the overall validation accuracies:

Optimizers: **[Adam, SGD]**

Learning Rates = **[1e-4, 1e-3, 1e-2, 1e-1]**

Batch Sizes = **[16, 32]**

Number of Epochs = **5**

The recommended ranges for learning rate and batch size were used. With respect to the optimizers, Adam was selected due to its adaptive learning mechanism and faster convergence, making it suitable for limited training epochs. SGD is also commonly used in image classification, as it often provides strong generalization when properly tuned. The best performing hyperparameter setting for each of the models are highlighted in the below table:

<img width="768" height="157" alt="image" src="https://github.com/user-attachments/assets/c3588564-f6e0-4ce6-8420-1ee0c7d2a789" />

Note: Hyperparameter tuning was initially performed using a reduced number of epochs (5 to be precise) to efficiently compare configurations. After selecting the best settings, the final model is trained for a higher number of epochs (10 to be precise) to ensure proper convergence and stable performance evaluation.

**<ins>Detailed Tuning Results:</ins>**

<img width="741" height="341" alt="image" src="https://github.com/user-attachments/assets/b5b0ce9c-f4d5-435f-89ee-b1ee6cd411b9" />
<img width="741" height="320" alt="image" src="https://github.com/user-attachments/assets/50ab0edf-a4ab-422a-98ff-ec94d018185f" />
<img width="741" height="319" alt="image" src="https://github.com/user-attachments/assets/21cfa5e0-1c6e-4a5a-8fc8-07923cfbe9d7" />


# MODEL EVALUATION

Model evaluation is performed exclusively on the test set to ensure an unbiased assessment of generalization performance. No additional training, hyperparameter tuning, or validation adjustments are conducted during this phase. The evaluation is carried out in inference mode, with gradients disabled and no weight updates applied, ensuring a pure forward-pass assessment of the trained models.

Performance metrics are computed using the weighted averaging method **(average = "weighted")**, which accounts for class imbalance by weighting each class according to its support. This approach is particularly appropriate for multi-class classification tasks such as the RealWaste dataset, as it provides a more representative overall performance measure.

Here is the test set performance for each of the baseline and primary model using the **“Adam”** optimizer:

<img width="890" height="250" alt="image" src="https://github.com/user-attachments/assets/b4e57096-6db5-401a-900b-519c9834f7ca" />

Here is the test set performance for each of the baseline and primary model using the “SGD” optimizer:

<img width="894" height="261" alt="image" src="https://github.com/user-attachments/assets/bcb60831-9559-43a2-855e-3289fba109d2" />

Clearly **ResNet18 with Adam optimizer outperformed** ResNet34 (with any optimizer) across all evaluation metrics. F1-score of 78.97 indicates the overall classification quality is solid and balanced.

**Note**: Although ResNet34 is a deeper architecture, ResNet18 achieved better generalization performance on the RealWaste dataset. This suggests that model complexity must be matched appropriately to dataset size and task complexity, as deeper networks may lead to overfitting when training data is limited.

Again, though based on the tuning results, EfficientNet-B0 (with SGD optimizer) was assumed to be best performing among all the models, the test set performance of EfficientNet-B0 model showed almost similar to the chosen primary model (Fig. 11).

<img width="451" height="269" alt="image" src="https://github.com/user-attachments/assets/28de2e5d-71ae-49a7-bc2f-534354973ff8" />


Higher values on the diagonal of the confusion matrix below (Fig. 12) indicates overall solid performance with the number of correctly classified samples for each class. **The best-classified classes are Plastic, Metal, Paper and Vegetation** show very high correct predictions with minimal confusion. Main confusions occur between similar materials such as Plastic vs Metal. Miscellaneous Trash is frequently confused with Plastic, Metal, and Textile Trash.

<img width="465" height="445" alt="image" src="https://github.com/user-attachments/assets/6ba7417e-c293-414d-8fd8-2f60b8d72412" />

**<ins>Detailed Evaluation Results with varied models and settings:</ins>**

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


# CONCLUSIONS

This project aimed to address the challenge of waste segregation by developing an automated image classification system, contributing to a significant sustainability issue. Improper waste sorting reduces recycling efficiency and increases environmental impact; therefore, intelligent automation can support more accurate and scalable waste management solutions. By applying computer vision and transfer learning, this work demonstrates a practical approach for automatic waste classification.

Three pre-trained Convolutional Neural Network (CNN) architectures such as ResNet34 as a baseline, ResNet18 as the primary model, and EfficientNet-B0 as the optimized model were evaluated and fine-tuned by experimenting with different optimizers (SGD and Adam), learning rates, batch sizes, and other hyperparameters. Among the tested models, ResNet18 and EfficientNet-B0 both achieved almost similar overall performance, confirming the effectiveness of transfer learning for this task.

The proposed system has strong potential for real-world deployment in automated sorting facilities or smart waste management systems, helping reduce manual effort and classification errors. All resources and code are made publicly available to encourage further research and improvement. Considering the best achieved accuracy of 78.99% (with the chosen primary model), immediate further work can be combining datasets such as RealWaste, TrashNet, and TrashBox with advanced data augmentation techniques and with additional regularization strategies to train the model again to achieve better accuracy. Other future work may include developing fully deployable real-time systems utilizing the saved trained model. While promising results have been achieved, continued research can further enhance intelligent waste segregation technologies. 


# SUPPLEMENTARY MATERIAL

Supplementary material belonging to this project can be accessed as follows:

**Dataset:**

https://www.kaggle.com/datasets/joebeachcapital/realwaste/data

Alternative link: https://github.com/sam-single/realwaste

**ImageNet-pretrained weights for:**

**ResNet18**: https://download.pytorch.org/models/resnet18-f37072fd.pth

**ResNet34**:https://download.pytorch.org/models/resnet34-b627a593.pth

**EfficientNet-B0**: https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth

**Trained Models:**

https://drive.google.com/drive/folders/1SIucKwq8YQzus_ybRob19HOHWgbNF05-?usp=drive_link


# REFERENCES

**RealWaste Dataset:**

https://www.mdpi.com/2078-2489/14/12/633

**Models and pre-trained weights:**

https://docs.pytorch.org/vision/0.12/models.html

**ResNet from Deep Residual Learning for Image Recognition:**

https://arxiv.org/abs/1512.03385

**PyTorch documentation of ResNet18:**

https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

**PyTorch documentation of ResNet34:**

https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html

**PyTorch documentation of EfficientNet-B0:**

https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html

**TrashNet Dataset (Gary Thung): ~2,500 labelled images across 6 classes:**

https://github.com/garythung/trashnet

**TrashBox: ~17,785 images divided into seven classes:**

https://github.com/nikhilvenkatkumsetty/TrashBox
