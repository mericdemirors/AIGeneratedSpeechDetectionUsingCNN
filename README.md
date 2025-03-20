# [AI Generated Speech Detection Using CNN](https://ieeexplore.ieee.org/abstract/document/10825541)

The recent rapid developments in generative-AI research has made it exceedingly hard to distinguish artificially generated audio-visual content from real ones. As a result, reliably detecting synthetic content has become an important problem to solve. In this study, multiple CNN, FC and SVM models are trained to detect synthetic audio signals obtained by using generative-AI models. The test results on realistic audio test sets show that the best accuracy scores for the CNN, FC and SVM models are 99.06, 99.15 and 98.68%, respectively. These results point out that the synthetic audio signals can be discriminated from the real ones by the trained models. Therefore, the proposed solution can be used in real-life practical applications to tackle this problem. Our analyses show that CNN models are the most suitable compared to other techniques, as FC and SVM models can also detect synthetic audios but have different inherent disadvantages.

This github repository is for people who wants to experiment with our results, try to improve them, or use our research as a base for their works.


## Method
We focused on features that capture high-frequency information, dynamic characteristics, and detailed spectral content. Bispectrum analysis was chosen over MFCC due to its visibly distinctive features in audio signals, which reveal higher-order correlations in the Fourier domain.

### Bispectrum Analysis
The bispectrum, calculated to capture third-order correlations, uncovers "unnatural" patterns that may indicate synthetic speech. To facilitate analysis, we use both the magnitude and phase components of the complex bispectrum. Additionally, we employ normalized bispectrum (or bicoherence) by segmenting the signal and averaging each segment’s spectrum.

### Feature Extraction Process
Audio data is converted into bispectrum images, from which we derive five specific features: absolute, angle, cum3, real, and imag. The audio is divided into segments to compute the bispectrum, with each feature computed per segment to capture unique aspects of the audio. These features are then normalized and processed into a "signature image," which serves as input for SVM, FC, and CNN models for classification.

## Models
### FC Models
FC models were trained on multiple processed input data types, including raw data, PCA-reduced features, and features extracted using ResNet50. A total of 24 different training configurations were explored, with six types of data for each training style. The models used a basic architecture with fully connected layers, ReLU activations, and sigmoid for the output layer. Models using dimensionality reduction or feature extraction had adjusted architectures to accommodate smaller input sizes and added hidden layers to balance the parameter count.

### CNN Models
Three CNN architectures were trained: ResNet50, GoogLeNet, and a custom basic_CNN. These models were trained on raw feature inputs, with variations including RGB-stacked features and pre-trained weights for the feature extractors. A total of 38 trainings were performed, including configurations with individual features and multi-form data inputs. All models were modified for binary classification.

### SVM Models
Two SVM models (Linear Support Vector Classification and C-Support Vector Classification) were trained on raw, PCA and UMAP-reduced data, as well as features extracted by ResNet50. A total of 72 trainings were performed, using a range of input data formats and dimensionalities, including 64 and 256 dimensions. Both SVM models were trained with configurations suitable for each type of input data.

## Dataset
The ['In-the-Wild' Dataset](https://deepfake-total.com/in_the_wild) was used for this research, which includes real and synthetic voice recordings of famous personalities. To ensure balanced class distribution for training, features were extracted from all 11,816 instances of both real and synthetic voices. The dataset was split into training, validation, and test sets with ratios of 0.8, 0.1, and 0.1, respectively.

## Conclusion
This research explores and compares various machine and deep learning approaches for detecting AI-generated audio, using CNNs, fully connected networks, and SVMs on image features derived from audio. The best model achieved an impressive accuracy of 99.15%. Among 134 training runs, most results achieved accuracy rates above 97%, with CNN models proving to be the most suitable overall for this detection task.

Future work will focus on identifying more distinguishing features and experimenting with new models, including vision transformers and MAMBA models, to detect synthetic segments within real audio. Once a robust detection model is developed, it will inform further improvements in generative model training.

## Paper and Citation
[Go to paper](materials/AI_Generated_Speech_Detection_Using_CNN.pdf)
```bibtex
"AI Generated Speech Detection Using CNN", Meriç Demirörs, Toygar Akgün and Ahmet Murat Özbayoğlu, 2024
```
