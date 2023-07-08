#   Importing the necessary modules/libraries.

from flask import Flask, render_template, request, send_from_directory
import os
import platform
import librosa
import numpy as np
import pandas as pd
import annoy
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

application=Flask(__name__, template_folder="templates", static_folder="static")    #   Creating a Flask object.
if platform.system()=="Windows":    #   Checking if the current operating system is a Windows one.
    application.config["UPLOAD_FOLDER"]=r"static\files"
else:
    application.config["UPLOAD_FOLDER"]=r"static/files"

#   Function to load the AnnoyIndex object.

def load_annoy_index():
    global annoy_index
    annoy_index=annoy.AnnoyIndex(40, "angular") #   Creating an AnnoyIndex object with the angular distance metric.
    annoy_index.load("music.ann")   #   Loading the AnnoyIndex object from the already stored file.
    
#   Function to load the features of the audio files.

def load_features():
    global feature_dataframe
    global feature_array
    feature_dataframe=pd.read_pickle("features.pkl")    #   Reading the pickle (.pkl) file into a pandas.DataFrame.
    feature_array=np.array(feature_dataframe.Feature.tolist())  #   Converting the features to a NumPy.NDArray.
    
#   Function to extract the Mel-Frequency Cepstral Coefficients (MFCC) features from the audio file.

def extract_features(file_name):
    audio, sample_rate=librosa.load(file_name, res_type="kaiser_fast")  #   Loading the audio file.
    mfcc=librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)   #   Extracting the Mel-Frequency Cepstral Coefficients (MFCC) features.
    mfcc_scaled=np.mean(mfcc.T, axis=0) #   Scaling the Mel-Frequency Cepstral Coefficients (MFCC) features.
    return mfcc_scaled

#   Function to return the approximate nearest neighbours of the recently read audio file.

def get_nearest_neighbours(new_audio, annoy_index, n_neighbours):
    new_audio_mfcc=extract_features(new_audio)  #   Extracting the Mel-Frequency Cepstral Coefficients (MFCC) features from the recently read audio file.
    nearest_neighbours=annoy_index.get_nns_by_vector(new_audio_mfcc, n_neighbours)  #   Retrieving the approximate nearest neighbours of the recently read audio file.
    return nearest_neighbours

#   Function to return the cosine distances between the recently read audio file and its approximate nearest neighbours.

def compare_vectors(new_mfcc, mfcc, nearest_neighbours):
    distances=[]
    for neighbour in nearest_neighbours:    #   Iterating through the approximate nearest neighbours of the recently read audio file.
        distances.append(cosine(new_mfcc, mfcc[neighbour])) #   Calculating the cosine distances between the recently read audio file and its approximate nearest neighbours.
    return distances

#   Function to return the best match (highest metric value) of the recently read audio file.

def get_best_match(new_audio, mfcc, annoy_index, n_neighbours):
    nearest_neighbours=get_nearest_neighbours(new_audio, annoy_index, n_neighbours) #   Retrieving the approximate nearest neighbours of the recently read audio file.
    distances=compare_vectors(extract_features(new_audio), mfcc, nearest_neighbours)    #   Calculating the cosine distances between the recently read audio file and its approximate nearest neighbours.
    best_match=nearest_neighbours[np.argmin(distances)] #   Retrieving the best match (highest metric value) of the recently read audio file.
    return best_match

#   Function to return the worst match (lowest metric value) of the recently read audio file.

def get_worst_match(new_audio, mfcc, annoy_index, n_neighbours):
    nearest_neighbours=get_nearest_neighbours(new_audio, annoy_index, n_neighbours) #   Retrieving the approximate nearest neighbours of the recently read audio file.
    distances=compare_vectors(extract_features(new_audio), mfcc, nearest_neighbours)    #   Calculating the cosine distances between the recently read audio file and its approximate nearest neighbours.
    worst_match=nearest_neighbours[np.argmax(distances)]    #   Retrieving the worst match (lowest metric value) of the recently read audio file.
    return worst_match

#   Function to perform audio segmentation using the K-means clustering algorithm on the recently read audio file.

def audio_segmentation(new_audio, annoy_index, n_neighbours, n_clusters):
    neighbours=get_nearest_neighbours(new_audio, annoy_index, n_neighbours) #   Retrieving the approximate nearest neighbours of the recently read audio file.
    kmeans=KMeans(n_clusters=n_clusters, random_state=0, algorithm="elkan").fit(feature_array[neighbours])  #   Performing K-means clustering on the approximate nearest neighbours of the recently read audio file.
    cluster_labels=kmeans.labels_   #   Retrieving the cluster labels of the approximate nearest neighbours of the recently read audio file.
    cluster_dataframe=pd.DataFrame({"Cluster": cluster_labels, "Song": feature_dataframe.Label[neighbours]})    #   Creating a pandas.DataFrame from the cluster labels and the corresponding song names.
    cluster_dataframe=cluster_dataframe.sort_values(by=["Cluster"]) #   Sorting the pandas.DataFrame by the cluster labels.
    cluster_dataframe=cluster_dataframe.reset_index(drop=True)  #   Resetting the index of the pandas.DataFrame.
    cluster_dataframe.to_csv("pied_piper_download.csv", index=False)    #   Saving the pandas.DataFrame into a separate comma-separated values (CSV) file.

load_annoy_index()
load_features()

#   Function to render the favicon.

@application.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(application.root_path, "static"),
                               "musical_icon.png", mimetype="image/vnd.microsoft.icon")

#   Function to render the home page.

@application.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

#   Function to render the recommendation page.

@application.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method=="POST":
        file=request.files["file"]  #   Retrieving the recently uploaded audio file.
        file_name=file.filename
        file_path=os.path.join(application.config["UPLOAD_FOLDER"], file_name)
        file.save(file_path)    #   Saving the recently uploaded audio file into the default folder.
        print(file_path)
        best_match=get_best_match(file_path, feature_array, annoy_index, 100)
        worst_match=get_worst_match(file_path, feature_array, annoy_index, 100)
        song_one_path=feature_dataframe.Label[best_match]
        song_two_path=feature_dataframe.Label[worst_match]
        audio_segmentation(file_path, annoy_index, 100, 10)
    return render_template("predict.html", song_one_name=song_one_path, song_one_path=song_one_path, song_two_name=song_two_path, song_two_path=song_two_path)

#   Driver function.

if __name__=="__main__":
    application.run(debug=True)