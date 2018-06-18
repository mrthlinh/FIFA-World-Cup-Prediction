from bs4 import BeautifulSoup
import requests
import pandas as pd
import sys
from bs4 import UnicodeDammit
from unidecode import unidecode

#url = "https://en.wikipedia.org/wiki/UEFA_Euro_2008_squads"
#url = "https://en.wikipedia.org/wiki/UEFA_Euro_2012_squads"
url =  "https://en.wikipedia.org/wiki/UEFA_Euro_2016_squads"
#url = "https://en.wikipedia.org/wiki/2017_FIFA_Confederations_Cup_squads"
#url = "https://en.wikipedia.org/wiki/2013_FIFA_Confederations_Cup_squads"
#url = "https://en.wikipedia.org/wiki/2009_FIFA_Confederations_Cup_squads"

#url = "https://en.wikipedia.org/wiki/Copa_Am%C3%A9rica_Centenario_squads"
# url = "https://en.wikipedia.org/wiki/2015_Copa_Am%C3%A9rica_squads"
#url = "https://en.wikipedia.org/wiki/2011_Copa_Am%C3%A9rica_squads"
#url = "https://en.wikipedia.org/wiki/2007_Copa_Am%C3%A9rica_squads"

#url = "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_squads"
#url = "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup_squads"
#url = "https://en.wikipedia.org/wiki/2010_FIFA_World_Cup_squads"
#url = "https://en.wikipedia.org/wiki/2006_FIFA_World_Cup_squads"
filename = url.split('/')[-1]
r = requests.get(url)

status = r.status_code
html_doc = r.text
soup = BeautifulSoup(html_doc,"lxml")

nations = soup.findAll('h3')
# EURO: 16
# WC: 32
num_nations = 24

arg_len = len(sys.argv)
if (arg_len > 1):
    num_nations = int(sys.argv[1])

tables = soup.findAll('table',class_="sortable")

squad_info = {}

i = 0
for nation in nations[0:num_nations]:
    country = nation.next_element.get_text()
    coach_name = nation.find_next_sibling('p').findAll('a')[-1].get_text()
    coach_name = unidecode(coach_name)

    # Alexandre Guimarães
    # print(unidecode(coach_name))
    print("{} - {}".format(country,coach_name))
    # try:
    #     print("{} - {}".format(country,coach_name))
    # except:
    #     print("{} - unicode map error".format(country))

    player_names = tables[i].select("th > a")

    rows = tables[i].select("tr")[1:]
    row_idx = 0
    for row in rows:
        _ = row.findAll('td')
        player_pos = _[1].find('a').get_text()
        player_name = row.select("th > a")[0].get_text()
        # player_name = player_names[row_idx].get_text()
        player_name = unidecode(player_name)
        # player_name = unidecode(player_names[i].encode("utf-8"))
        player_cap = _[3].get_text()

        player_info = {}
        player_info['Country'] = country
        player_info['Coach'] = coach_name
        player_info['player_pos'] = player_pos
        player_info['player_name'] = player_name
        player_info['player_cap'] = player_cap

        new_idx = len(squad_info)
        squad_info[new_idx] = player_info

        # player_goal= _[4].get_text()
        # print("{} - {}".format(player_pos,player_cap))
        print("{} - {} - {}".format(player_pos,player_name,player_cap))
        row_idx += 1
    # player_pos  = tables[i].select("td > a")[::2]  #keep even index
    # player_caps = tabl
    i += 1

    # print(name," - ",coach_name)
my_df = pd.DataFrame.from_dict(squad_info,orient='index')
my_df.to_csv(filename+'.csv', index = False)
