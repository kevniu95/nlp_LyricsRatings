from album_loader import load_saved_albums, Album, Albums, Song

class LyricProcessor():
    """
    Takes set of specifications for how to treat lyrics including:
        1. See / don't see parts
            - e.g., [CHORUS]
        2. Standardize parts
            -e.g., [Verse 1] -> [PART]
                    AND
                    [Pre-Chorus] -> [PART]
        3. See line breaks
            - <lb> and </lb>, respectively, would denote start and end of line
    
    -process_song() method is used to process song lyrics according to 
        instantiated specifications above
    """
    def __init__(self, see_parts = False, standardize_parts = False, see_line_breaks = True):
        self.see_parts = see_parts
        self.standardize_parts = standardize_parts  # Will override see_parts if set to True
        self.see_line_breaks = see_line_breaks
        
    def process_song(self, song):
        new_lyrics = []
        for line in song.lyrics:
            line = self._process_parts(line.strip())
            if len(line) > 0:
                new_lyrics.append(line)
        new_lyrics = self._process_linebreaks(new_lyrics)
        return new_lyrics

    def _process_parts(self, line):
        """
        Defines how to handle sections in lyrics that look like...
            - [Part I] or
            - [Intro]
        """
        if line.startswith('[') and line.endswith(']'):
            new_line = line.lower().replace(' ', '') if self.see_parts else ''
            new_line = '[part]' if self.standardize_parts else new_line
            return new_line
        return line

    def _process_linebreaks(self, lyrics):
        new_lyrics = lyrics
        if self.see_line_breaks and len(lyrics) > 0:
            new_lyrics = []
            length = len(lyrics)
            for num, line in enumerate(lyrics):
                if num < length:
                    # print(line)
                    new_line = '<lb> ' + line + ' </lb>'
                    new_lyrics.append(new_line)
                else:
                    new_lyrics.append(line)
        return new_lyrics

class RegAlbums():
    """
    Creates and stores the set of Albums used for Regression purposes
    Two important functions include...
        1. Limiting the original set of Albums based on 
            - a. Which albums have been "finalized"
            - b. Minimum number of critic ratings
            - c. Minimum number of user ratings
            - d. Minimum text length
        2. Pre-process Album text to potentially: 
            - a. Acknowledge / don't acknowledge parts
                -e.g., [Verse 1]
            - b. Standardize parts
                -e.g., [Verse 1] -> [PART]
                        AND
                       [Pre-Chorus] -> [PART]
            - c. See line breaks
                    - <lb> and </lb>, respectively, would denote start and end of line
            - d. See song breaks
                    - <sb> and </sb>, respectively, would denote start and end of song
    """
    def __init__(self, album_path = None, 
                        c_rate_min = 7, 
                        u_rate_min = 15, 
                        min_text_len = 200,
                        see_parts = True, 
                        standardize_parts = True,
                        see_line_breaks = True,
                        see_song_breaks = True):

        self.lyricProc = LyricProcessor(see_parts = see_parts, 
                                        standardize_parts = standardize_parts, 
                                        see_line_breaks = see_line_breaks)
        self.albums = {}
        if album_path:
            self.albums = load_saved_albums(album_path)
        self.see_song_breaks = see_song_breaks
        self.c_rate_min = c_rate_min
        self.u_rate_min = u_rate_min
        self.min_text_len = min_text_len
    
    def reg_full_album_text(self):
        init_albums = self._select_albums_for_reg()
        print(f"After making limitations, working with {len(init_albums)} albums in total...")
        
        reg_albums = []

        for num, album in enumerate(init_albums[:]):
            alb_text = self._get_album_text(album)
            alb_text = self._clean_tokens(alb_text)
            reg_albums.append((album.name, album.c_score, album.u_score, alb_text))
        
        final_reg_albums = [album for album in reg_albums if len(album[3]) >= self.min_text_len]
        share = len(final_reg_albums) / len(reg_albums) * 100
        print(f"{len(final_reg_albums)}/{len(reg_albums)} ({share:.1f}%) albums, have length >{self.min_text_len} and are retained.")
        return reg_albums

    def _select_albums_for_reg(self):
        initAlbums = []
        all_sum = 0
        sel_sum = 0
        for yr, albums in self.albums.items():
            all_ctr =0
            sel_ctr = 0
            for k, album in albums.items():
                all_ctr += 1
                try:
                    u_ratings = int(album.u_ratings)
                except:
                    u_ratings = -1
                c_ratings = int(album.c_ratings) if album.c_ratings else 0
                if album.finalized and u_ratings > self.u_rate_min and c_ratings > self.c_rate_min:
                    initAlbums.append(album)
                    sel_ctr += 1
            # print(f"In year {yr}, there are {all_ctr} rated albums. Selected: ")
            # print(f"  -{sel_ctr} ({100 * sel_ctr / all_ctr :.1f}%) ")
            all_sum += all_ctr
            sel_sum += sel_ctr
            # print(yr, all_ctr, sel_ctr)
        
        # print(f"All sum: {all_sum}")
        # print(f"Sel sum: {sel_sum}")
        # print(f"Selected portion: {100 * (sel_sum / all_sum) :.1f}%")
        return initAlbums
    
    def _clean_tokens(self, text):
        """
        Some symbols retrieved from Genius should be replaced
        with ASCII equivalents
        """
        TOKENS = {'’' : "'", '‘' : "'"}
        for k, v in TOKENS.items():
            text = text.replace(k, v)   
        return text

    def _get_album_text(self, album):
        """
        For each song in album, pass to Lyric Processor for pre-processing
        """
        album_lyrics_whole = []
        for song in album.songs:
            song_lyrics = self.lyricProc.process_song(song)
            album_lyrics_whole.append(song_lyrics)
        if self.see_song_breaks:
            album_lyrics_whole = self._process_songbreaks(album_lyrics_whole)
        list_of_lines = [x for xs in album_lyrics_whole for x in xs]
        return ' '.join(list_of_lines)
    
    def _process_songbreaks(self, songs):
        if len(songs) > 0:
            for song in songs:
                song.insert(0, '<sb>')
                song.append('</sb>')
        return songs

    def test_album_text(self, alb_text):
        for num, i in enumerate(alb_text):
            if alb_text[num - 5 : num ] in ['</sb>']:
                print()
            if alb_text[num : num + 4] in ['<sb>']:
                print()
            print(i, end = '')
            if alb_text[num-4 : num] in ['<sb>'] or alb_text[num-5 : num ] in ['</lb>']:
                print()
        print()

def main():
    a = RegAlbums(album_path = 'albums_f.pickle')
    all_albs = a.reg_full_album_text() 
    # List of tuples
    # Each tuple: (Album Name, Metascore (critic), User Score, Lyrics Text)
    
    print(f"Will use {len(all_albs)} observations in total for modeling.")
    print()
    print("Here is an example of what an observation returned by reg_ful_album_text looks like")
    print("(Note this is used as input to PyTorch models later on...)")
    last_obs = all_albs[-1]
    print(f"First item in tuple:\n{last_obs[0]}")
    print(f"Second item in tuple:\n{last_obs[1]}")
    print(f"Third item in tuple:\n{last_obs[2]}")
    print(f"Last item in tuple (first 2500 characters):\n{last_obs[3][:2500]}")
    

if __name__ == '__main__':
    main()