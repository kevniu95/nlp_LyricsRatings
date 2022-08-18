import pickle
from selenium.webdriver.common.by import By

class AlbumLinkLoader:
    """
    1. Saves full set of album links from metacritic to 'links' property
        Ex. 
            { 2021 : [
                    'https://www.metacritic.com/music/conflict-of-interest/ghetts', 
                    'https://www.metacritic.com/music/prioritise-pleasure/self-esteem',
                    'https://www.metacritic.com/music/glow-on/turnstile', 
                    'https://www.metacritic.com/music/were-all-alone-in-this-together/dave',
                    'https://www.metacritic.com/music/carnage/nick-cave-warren-ellis']
                    ...
                    ],
             ...
            }

    """
    def __init__(self, pickle_file, driver):
        self.mc_link = "https://www.metacritic.com/browse/albums/score/metascore/year/filtered?year_selected={}&distribution=&sort=desc&view=detailed"
        self.pickle_file = pickle_file
        self.links = None
        try:
            with open(pickle_file, 'rb') as handle:
                self.links  = pickle.load(handle)
        except:
            print(f"Could not find {pickle_file}, generating new set of album links...")
            self.update_album_links(driver)
            with open(pickle_file, 'rb') as handle:
                self.links  = pickle.load(handle)
    
    def print_links(self):
        """
        Prints the links associated with each year
        """
        for k, v in self.links.items():
            print(k)
            print(v)
            print('\n')

    def update_album_links(self, driver):
        """
        For each year, obtains albums on Metacritic with at least 7 critic ratings

        kwargs:
        driver -- Selenium webdriver
        """
        if self.links is None:
            album_links = {}
        else:
            with open(self.pickle_file, 'rb') as handle:
                album_links = pickle.load(handle)
            
        for i in range(2006, 2022):
            print(i)
            if i in album_links.keys() and len(album_links[i]) > 0:
                print("This year already has data, continuing...")
                continue
            this_yr_links = []
            driver.get(self.mc_link.format(str(i)))

            this_yr_links = self.get_titles(driver, this_yr_links)

            pages = driver.find_elements(By.CLASS_NAME, 'page_num')
            page_links = [page.get_attribute('href') for page in pages]
            
            ctr = 0
            for page in page_links:
                print(ctr)
                if page is not None:
                    driver.get(page)
                    this_yr_links = self.get_titles(driver, this_yr_links)
                ctr += 1
            this_yr_links = [i for i in this_yr_links if i is not None]
            album_links[i] = this_yr_links
            print()
            
        with open(self.pickle_file, 'wb') as handle:
            pickle.dump(album_links, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.links = album_links
        
    def get_titles(self, driver, this_yr_links):
        """
        Gets link for this album

        kwargs:
        driver -- Selenium web driver
        this_yr_links -- list of links to be updated
        """
        titles = driver.find_elements(By.CLASS_NAME, "title")
        page_album_links = [title.get_attribute('href') for title in titles if title is not None]
        this_yr_links = this_yr_links + page_album_links
        return this_yr_links