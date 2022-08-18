# nlp_LyricsRatings

## 1. Data Preparation:
- *albumLinks.py*
  - Starting point for obtaining albums data
  - Obtains Metacritic.com album links to all albums with at least 7 critic reviews from 2006 - 2021
  - Saves to album_links.pickle
- *album_loader.py*
  - Gets all Metacritic and Genius information associated with albums
  - Saves results to albums_f.pickle
- *lyric_loader.py*
  - Retrieves saved lyrics from albums_f.pickle and pre-processes them to be ready to serve as input in regression tasks
  - RegAlbums() class is bridge between accumulated pre-processing work and rest of ML pipeline downstream
  
## 2. Non-BERT pipeline:
- *nlpmodel.py*
  - Houses all different collate functions that may be required by various models/methodologies
  - Also houses the actual PyTorch modules for models/methodologies
  - Provides generic functions and class - NlpModel() - to run based on model details specified in *generalRunner* files
- *generalRunner.py* and *generalRunner.ipynb*
  - Prepares data for regression, trains model, evaluates results, and saves results based on model specified at command line 
  (for .py) or bottom-most cell (for.ipynb)

## 3. BERT
- *BERT fine-tune.ipynb*
  - Sets up, trains, and evaluates BERT-based model fine-tuned for this regression task
  
  
  
  
  

