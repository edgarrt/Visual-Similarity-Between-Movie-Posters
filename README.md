# Final Project: Visual Similarity Between Movie Posters

## Implementation of [Classifying_Movie_Poster_Genres_with_Deep_Metric_Learning.pdf](https://github.com/edgarrt/Visual-Similarity-Between-Movie-Posters/blob/master/report/Classifying_Movie_Poster_Genres_with_Deep_Metric_Learning.pdf)


---

## Project Highlights
- leverage a base ResNet model combined with proxy anchor loss to learn an embedding space for movie posters.

## Previous Work
Research draws upon previous work in convolutional neural networks, movie genre classification, and metric learning. The ResNet architecture, proxy anchor loss, and deep metric learning methods are some of the notable techniques utilized.

## Technical Approach
The following key steps make up the project workflow:
- Data Collection and Preprocessing: Movie poster dataset collected from themoviedb.org and preprocessed for uniform size, normalization, and data augmentation.
- Network Architecture: Using ResNet50 as the backbone for our embedding model.
- Learning Method: Employing the Proxy Anchor Loss to train the ResNet50-based embedding model.
- Model Training: Utilizing PyTorch and PyTorch Lightning for training.
- Evaluation and Testing: Evaluating the model with Recall@K metric.
- Similar Movie Posters Retrieval: Using Approximate Nearest Neighbors (ANN) to retrieve similar movie posters.

## Dataset and Implementation
Created a new movie dataset by leveraging themoviedb.org's API. This decision was because existing public movie datasets did not contain recent movie releases and had issues with acquiring the movie poster image files. The API provided data for 10,000 popular movies.


## Getting started with development

## Prereqs:
- Python 3.10+

## Check if you have Python 3.10 installed
`python3.10 --version`

If you get
`python3.10: command not found`

try:
`python3 --version`

If you get 
`Python 3.X.Y` where `X` is at least 10 then update the last line in the `Pipfile` to your output and continue to next section

ex:
```
main@LAPTOP:~/cosc525_final_project$ python3 --version
Python 3.10.6

# Pipfile
[requires]
python_version = "3.10"

```

Follow: [Installing Python 3.X](https://www.codingforentrepreneurs.com/blog/install-django-on-mac-or-linux/), but install the Python 3.10.6 binary

Should be this page - https://www.python.org/downloads/release/python-3106/




## Python Virtual Env
Create the virtual environment & install the requirements:
```
# this installs numpy automatically
pipenv install --dev

```
Enter the virtual environment:
```
pipenv shell
```



## Launch Jupyter notebook environment (Mostly for local development):
```
# This launches the notebook env & should open your browser
jupyter notebook
```

## License
Visual Similarity Between Movie Posters is available under the MIT license. See the LICENSE file for more info.

## Author
Developed by [Edgar Trujillo](https://www.linkedin.com/in/trujilloedgar/)