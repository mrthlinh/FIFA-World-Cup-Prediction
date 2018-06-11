# Scrap web
from bs4 import BeautifulSoup
import requests
import pandas as pd
import sys
from unidecode import unidecode

def retrieve_info(r,verbose=False):
    global df_index
    global squad_info
    html_doc = r.text
    soup = BeautifulSoup(html_doc,"lxml")


    # Find the table of players
    table = soup.find('table', class_='table table-striped players')
    # Extract url to player info
    player_link = soup.find_all('td', attrs={"data-title": "Name"})
    # Build list of full url
    inner_link = [url_home+link.a['href'] for link in player_link] # List compprehension

    # Iterate through the list of player and get info
    for link in inner_link:
        progress = "\r"+str(inner_link.index(link)+1) + "/" + str(len(inner_link))
        print(progress,end = ' ')

        sub_html_doc = requests.get(link).text
        sub_soup = BeautifulSoup(sub_html_doc,"lxml")


        tables = sub_soup.findAll('div', attrs={"class": "panel panel-info"})
        stars = sub_soup.findAll('span', class_ = "star")
        i = 0
        player_info = {}
        country = sub_soup.find('h2',attrs={"class": "subtitle"}).get_text()
        player_info['Country'] = country
        for table in tables:

            suffix = ""
            header = table.find('h3',attrs={"class": "panel-title"}).get_text()
            if (i == 0):
                _ = header.split(" ")
                # player_info['Name'] = ' '.join(_[0:-2])
                player_name = ' '.join(_[0:-2])
                player_name = unidecode(player_name)
                player_info['Name'] = player_name

                player_info['OverRate'] = _[-2]
                player_info['PotRate'] = _[-1]
            elif ((i == 1) & (len(tables) >= 9)):
                player_info['Club'] = header.strip()
                suffix = "Club_"
            elif ((i == 2) & (len(tables) == 10)):
        #        country = table.find('h3',attrs={"class": "panel-title"}).get_text()
#                player_info['Country'] = header
                suffix = "Nation_"
            else:
                suffix = header.replace(" ","") + "_"

            body = table.find('div',attrs={"class": "panel-body"}).findAll('p')

            for row in body:
    #            print(suffix,row.contents[0],":",row.contents[1].get_text("-"))
                feature_name = suffix + row.contents[0].strip().replace(" ","")
                value = row.contents[1].get_text("-")
                if (len(stars) >= 1):
                    if (feature_name == 'WeakFoot'):
                        value = len(stars[0].findAll('i',class_="fa fa-star fa-lg"))

                    if (feature_name == 'SkillMoves'):
                        value = len(stars[1].findAll('i',class_="fa fa-star fa-lg"))
#                else:
#                    value = row.contents[1].get_text("-")
                if ((feature_name == 'Height') | (feature_name == 'Weight')):
                    _ = value.split(" ")[0]
                    value = _
                player_info[feature_name] = value

            i += 1

        if (verbose):
           print(player_info['Name'])

        squad_info[df_index] = player_info
        df_index += 1

# ==================================================================

#print ("This is the name of the script: ", sys.argv[0])
# print ("Number of arguments: ", len(sys.argv))
# print ("The arguments are: " , str(sys.argv))

url_home = 'https://www.fifaindex.com'
versions = ['fifa05_1','fifa06_2','fifa07_3','fifa08_4','fifa09_5','fifa10_6',
'fifa11_7','fifa12_9','fifa13_11','fifa14_13','fifa15_14','fifa16_73','fifa17_173','']

#version_id = 0
version_id = int(sys.argv[1]) - 5
version = versions[version_id]
print("Version of FIFA: ",version)

start_page = 1
debug = False


if len(sys.argv) >=3:
    start_page = int(sys.argv[2])
if len(sys.argv) >=4:
    if (sys.argv[3] == 'T'):
        debug = True

df_index=0
squad_info = {}
next_page = start_page
while True:
#        url = url_home + '/players/'+version+'/'+str(next_page)+'/?nationality='+str(next_nation)
    url = url_home + '/players/'+version+'/'+str(next_page)
    try:
        r = requests.get(url)
        status = r.status_code
        if (status != 200):
            break
        print("\npage ",next_page)
        retrieve_info(r, verbose=debug)
        next_page += 1
    except:
        break
#        test += 1
    if debug:
        if (next_page == 2):
            break

#    next_nation += 1

my_df = pd.DataFrame.from_dict(squad_info,orient='index')
my_df.to_csv(version+'.csv', index = False)
