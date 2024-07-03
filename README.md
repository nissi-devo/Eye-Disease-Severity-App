# Eye-Disease-Severity-App

<h2><strong>Overview</strong></h2>
Evaluating the severity of eye diseases using medical images is an essential and routine task in medical diagnosis and treatment. Traditional grading systems based on discrete classification are often unreliable and fail to capture the entire spectrum of eye disease severity. This project aims to address these issues by developing a system that determines the severity of eye diseases on a continuous scale using a twin-convolutional neural network approach known as Siamese Neural Networks. This system is particularly demonstrated in the domain of diabetic retinopathy.

<h2><strong>Features</strong></h2>

**Upload Retinal Fundus Images:** Users can upload retinal fundus images to the web application.

**Disease Severity Scoring:** The application processes the images using a Siamese Neural Network to generate a continuous severity score.

**Comparison Analysis:** The system compares the uploaded image with reference images to determine the severity score based on image embedding distances.

**User-Friendly Interface:** A simple and intuitive web interface for easy interaction.

<h2><strong>Siamese Neural Networks for Eye Disease Severity</strong></h2>
The Siamese Neural Network approach leverages twin convolutional neural networks to compute the similarity between pairs of images. In this application, we use a Siamese Triplet network to find the distance between image embeddings. The system evaluates the performance using samples of retinal fundus images from an eye clinic in India. The outputs show a positive correlation (95%) with originally assigned severity classes, indicating a continuous range of severity and changes in eye diseases.
