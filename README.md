# Machine Learning Papers Summary

Docsify Book: https://gitycc.github.io/machine-learning-papers-summary  
Github Repo: https://github.com/GitYCC/machine-learning-papers-summary

### Motivation

Reading papers is important because ML/DL/AI is the field of continuous innovation. This repository chose most cited ML/DL/AI papers to read and summary.

### Paper Survey

- [https://github.com/terryum/awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers)

### Content

#### Understanding / Generalization / Transfer

- **Distilling the knowledge in a neural network** \(2015\), G. Hinton et al. \[➤ [s](understanding-generalization-transfer/distilling-the-knowledge-in-a-neural-network.md)[ummary](understanding-generalization-transfer/distilling-the-knowledge-in-a-neural-network.md)\]
- **Deep neural networks are easily fooled: High confidence predictions for unrecognizable images** \(2015\), A. Nguyen et al. \[➤ [summary](understanding-generalization-transfer/deep-neural-networks-are-easily-fooled-high-confidence-predictions-for-unrecognizable-images.md)\]
- **How transferable are features in deep neural networks?** \(2014\), J. Yosinski et al. \[➤ [summary](understanding-generalization-transfer/how-transferable-are-features-in-deep-neural-networks.md)\]
- **CNN features off-the-Shelf: An astounding baseline for recognition** \(2014\), A. Razavian et al. \[[paper](http://www.cv-foundation.org//openaccess/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf)\]
- **Learning and transferring mid-Level image representations using convolutional neural networks** \(2014\), M. Oquab et al. \[[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)\]
- **Visualizing and understanding convolutional networks** \(2014\), M. Zeiler and R. Fergus \[[paper](http://arxiv.org/pdf/1311.2901)\]
- **Decaf: A deep convolutional activation feature for generic visual recognition** \(2014\), J. Donahue et al. \[[paper](http://arxiv.org/pdf/1310.1531)\]

#### Optimization / Training Techniques



#### Unsupervised / Generative Models



#### Convolutional Neural Network Models



#### Image: Segmentation / Object Detection



#### Image / Video / Etc



#### Advertising / Commerce

- **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction** (2017), H. Guo et al. \[➤ [summary](advertising-commerce/deepfm.md)\]



#### Natural Language Processing / RNNs



#### Speech / Other Domain



#### Reinforcement Learning / Robotics



### Contributors

-  [@GitYCC](https://github.com/GitYCC) \[website: [www.ycc.idv.tw](https://www.ycc.idv.tw), email: [ycc.tw.email@gmail.com](mailto:%20ycc.tw.email@gmail.com)\]



Welcome to contribute this repo. together.

### Contribution Rules

- Please use pull-request to merge new changes in `feature/xxx` branch into `master` branch
- Add new article
  - In `README.md`, choose or create a category, add new paper item (sorted by years decreasing) and add link `[➤ summary]` to your new article.
  - In `summary.md`, add path of your article into sidebar.
  - Please follow this file structure: (ref: https://docsify.now.sh)
    ```
    README.md
    summary.md
    some-category ---- assets
                   |     |--- some-image-1
                   |     |--- your-image-2
                   |-- your-article.md
    ```
  - In `your-article.md`, add contributor, paper or code at head of the article.

