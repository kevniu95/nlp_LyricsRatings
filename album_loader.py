from xml.sax.handler import property_declaration_handler
import sys
import requests
import pickle
import uuid
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from albumLinks import AlbumLinkLoader

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import pickle5 as pickle5


class Song():
    def __init__(self, name, album, lyric_link = None):
        self.name = name
        self.album = album
        self.lyric_link = lyric_link
        self.lyrics = None
        self.lyrics_set = False
       
    def get_lyrics(self):
        if self.lyric_link:
            print(self.lyric_link)
            page = requests.get(self.lyric_link)
            if page.status_code == 200:
                soup = BeautifulSoup(page.content, 'html.parser')
                delimiter = '###'
                display_blocks = soup.find_all('div', class_= re.compile('Lyrics__Container*'))
                strings = []
                if display_blocks is not None and len(display_blocks) > 0:
                    for display_block in display_blocks:
                        for line_break in display_block.findAll('br'):
                            line_break.replaceWith(delimiter)
                        strings = strings + display_block.get_text().split(delimiter)
                self.lyrics = strings
                self.lyrics_set = True
                return strings
            else:
                print("Unable to connect")                    
        else:
            print("Unable to connect")

    def __str__(self):
        return f"{self.album} - {self.name}"
    
    def __repr__(self):
        return f"{self.album} - {self.name}"

class Album():
    """
    Class representing a single Album
    Stores, among other things:
        1. Album name
        2. Artist
        3. Critic and User scores / ratings 
        4. Songs with lyrics
        5. Link to genius website
        6. 'Finalized' and 'lyrics set' statuses
    """
    def __init__(self):
        self.name = None
        self.id = uuid.uuid4()
        self.artist = None
        self.c_score = None
        self.c_ratings = None
        self.u_score = None
        self.u_ratings = None
        self.songs = []
        self.genius_link = "https://genius.com/albums/{}/{}"
        self.lyrics_tried = False
        self._finalized = False
    
    def __str__(self):
        return f"Album: {self.name}\nLyrics Tried: {self.lyrics_tried}\nInfo Finalized: {self.finalized}"

    def __repr__(self):
        return f"Album: {self.name}\nLyrics Tried: {self.lyrics_tried}\nInfo Finalized: {self.finalized}"

    @property
    def finalized(self):
        for song in self.songs:
            if not song.lyrics_set:
                self._finalized = False
                return self._finalized
        if (self.name and self.artist and self.c_score and self.c_ratings and 
                    self.u_score and self.u_ratings and len(self.songs) > 0):
            self._finalized = True
        else:
            self._finalized = False
        return self._finalized

    @property
    def lyrics_set(self):
        if len(self.songs) == 0:
            return False
        for song in self.songs:
            if not song.lyrics_set:
                return False
        return True

    def get_info(self, link, driver):
        """
        Function that collects:
        
        1. Album name and Artist
        2. Ratings information from Metacritic
        3. Lyrics information from Genius

        and stores as properties of this Album object
        """

        print("Getting album, artist name")
        # 1. Fill album name and artist, given metacritic link
        album, artist = link.split('/')[-2], link.split('/')[-1]
        self.name = ' '.join(album.split('-')).title()
        self.artist = ' '.join(artist.split('-')).title()

        # 2. Instantiate songs through genius
            # If can't do so, don't bother with rating
        print("Trying genius link...")
        try_link = self.genius_link.format(artist, album)
        self.songs = self._inst_genius_songs(try_link)
        if len(self.songs) == 0:
            print("Couldn't establish link to Genius lyrics website"\
                    " so will not process this album's ratings either")
            return
        
        # 3. Get Ratings from Metacritic
        print("Getting ratings")
        self.c_score, self.c_ratings, self.u_score, self.u_ratings = self._get_mc_rating(link, driver)

        # 4. Getting lyrics from Genius
        print("Getting lyrics")
        self.get_lyrics_for_songs()

    def _get_mc_rating(self, link, driver):
        try:
            driver.get(link)
            c_score = driver.find_element(By.XPATH, "//span[contains(@itemprop, 'ratingValue')]").text
        except:
            print("Ratings not properly loaded for this album!")
            return None, None, None, None
        c_ratings = driver.find_element(By.XPATH, "//span[contains(@itemprop, 'reviewCount')]").text
        
        u_score = driver.find_element(By.XPATH, "//div[starts-with(@class, 'metascore_w user')]").text
        u_ratings_cands = driver.find_elements(By.XPATH, "//a[contains(@href, 'user-reviews')]")
        u_ratings = u_ratings_cands[2].text.split(' ')[0]
        return c_score, c_ratings, u_score, u_ratings

    def _inst_genius_songs(self, link):
        songsList = []
        page = requests.get(link)
        if page.status_code == 200:
            soup = BeautifulSoup(page.content, 'html.parser')
            for display_block in soup.find_all('a', class_= 'u-display_block'):
                lyric_link = display_block.get('href')
                name = display_block.find('h3').text.strip().split('\n')[0]
                songsList.append(Song(name, self.name, lyric_link))
        else:
            print(f"Unable to connect to genius website for album {self.name}")
        return songsList
            
    def get_lyrics_for_songs(self):
        """
        Each song is instance of Song object with get_lyrics() from Genius method
        """
        for song in self.songs:
            song.get_lyrics()
        self.lyrics_tried = True


def load_saved_albums(album_path):
    if IN_COLAB:
        try: 
            with open(album_path, "rb") as fh:
                albums = pickle5.load(fh)
        except Exception as e:
            print(f"Could not find albums at {album_path}. Saving new Albums object at specified path")
            print("If you meant to load a pre-existing Albums object, please re-check file path.")
            albums = {}
    else:
        try:
            with open(album_path, 'rb') as handle:
                print(f"Found albums available at {album_path}. Will now load")
                albums = pickle.load(handle)
        except Exception as e:
            print(e)
            print(f"Could not find albums at {album_path}. Saving new Albums object at specified path")
            print("If you meant to load a pre-existing Albums object, please re-check file path.")
            albums = {}
    return albums

class Albums():
    def __init__(self, links = None, album_path = None):
        self._links = links # same as 'links' property from AlbumLoader class, 
                                # contains metacritic link for each album
        self.albums = {}
        self.album_path = album_path
        self.albums = load_saved_albums(album_path)
        
        if len(self.albums) == 0:
            self._inst_albums_from_links()
    
    def _inst_albums_from_links(self):
        """
        Instantiate Album instances in self.albums. Album instance indicates if:
            - Lyrics have been obtained
            - Ratings have been Obtained
            - Processing of Album has been finalized
        
        Albums are stored in dict of format:
            {album_metacritic_link : Album object}
        """
        print("\nInstantiating Album instances given the links provided...")
        for k, v in self.links.items():
            self.albums[k] = {}
            for i in v:
                self.albums[k][i] = Album()
        self._save_albums()
    
    def _save_albums(self, path = None):
        if path is None:
            path = self.album_path
        with open(path, 'wb') as handle:
            pickle.dump(self.albums, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved recently-updated albums out to {path}")

    def get_lyrics_for_year(self, year, driver, save = False):
        """
        Save lyrics for each album
        """
        yr_albums = self.albums[year]
        ctr = 0
        for album_link, album_object in yr_albums.items():
            print(album_link)
            ctr += 1
            if album_object.finalized:
                print(f"Lyrics are set for {album_object.name}")
            elif album_object.lyrics_tried:
                print(f"Tried unsuccessfully to get lyrics for {album_object.name}. Moving on")
            else:
                print("Getting metacritic info for this album...")
                album_object.get_info(album_link, driver)
                print(f"This album is finalized: {album_object.finalized}")
                print()
            if ctr % 10 == 0 and save:
                self._save_albums(save)
                print(f"Saved last 10 albums info to {save}")

    @property
    def links(self):
        if isinstance(self._links, str):
            try:
                with open(self._links, 'rb') as handle:
                    self._links  = pickle.load(handle)
            except:
                print(f"Could not find album links at {self._links}."\
                " Please supply links directly or provide another file path.")
                return
        if isinstance(self._links, dict):
            return self._links
        else:
            print("Please make sure links supplied are either a dictionary"\
            " of album links by year, or a filepath to such a dictionary")
            return


def folklore_check(driver):
    a = load_saved_albums('albums_f.pickle')
    folklore = [(folk_link, folk_Album) for folk_link, folk_Album in a[2020].items() if 'folklore' in folk_link][0]
    print(folklore)
    link, alb = folklore
    alb.get_info(link, driver)
    for song in alb.songs[:1]:
        print(song.lyrics)

def main():
    """
    1. Instantiating AlbumLinkLoader() gets metacritic link for all  
        - In this case, saves to 'album_links.pickle'

    2. Instantiating Albums() converts all links into objects of Album type
        and stores them in one place

    3. get_lyrics_for_year() method gets actual information for each Album, where available
        including...
        a. Album name and artist
        b. Ratings information from metacritic
        c. Lyrics information from Genius

        results are saved to 'albums_f.pickle in this case
    """
    get_driver = True
    if get_driver:
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = options)
    
    ALBUMS_LINKS = 'album_links.pickle'
    ALBUMS_FINALIZED = 'albums_f.pickle'

    AlbumLinkLoader(ALBUMS_LINKS, driver)
    
    albums = Albums(links = ALBUMS_LINKS, album_path = ALBUMS_FINALIZED)
    
    for k in range(2021, 2005, -1):
        albums.get_lyrics_for_year(k, driver, save = ALBUMS_FINALIZED)
        #Please note that the finalized, 'albums_f.pickle' file is saved in directory now.
    
    # folklore_check(driver)    

    driver.close()


if __name__ == '__main__':
    main()

