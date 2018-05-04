# Project Description
__Objective__: Prediction of the winner of FIFA world cup 2018. Prediction results are  Win / Lose / Draw 

__Data__: Data are assembled from multiple sources, most of them are frome Kaggle, others come from FIFA website / EA games and I need to build a data crawler.
 
__Feature Selection__: To determine who will more likely to win a match, based on my knowledge, I come up with 4 main groups of features as follows:
1. head-to-head match history
2. recent performance of each team (10 recent matches), aka "form"
3. bet-ratio before matches
4. squad value or how many players are in top 200 (from video game)
Feature list reflects those factors. 

__Lifecycle__
Some images here

# Data
### Data Source
The dataset are from all international matches from 2000 - 2017, results, bet odds, ranking, title won.
1. [FIFA World Cup 2018](https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data)
2. [International match 1872 - 2017](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data)
3. [Ranking through Time](http://www.fifa.com/fifa-world-ranking/associations/association=usa/men/index.html)
4. [Bet Odd](https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data)
5. [Player Rating Through time](https://www.futhead.com/10/players/?page=2)
6. [Squad for each tournament](https://github.com/openfootball/world-cup)

### Feature List

| Feature Name  | Group | Description              | Source | Status |
|:-------------:|:-----:|:------------------------:|:------:|:------:|
| team_1        |   N/A |Nation Code (e.g US, NZ)      |   2    |Done|
| team_2        |   N/A |Nation Code  (e.g US, NZ)     |   2    |Done|
| date          |   N/A |Date of match yyyy - mm - dd  |   2    |Done|
| home_team     |   N/A |Who is the home team          |   2    |Done|
| tournament    |   N/A |Friendly,EURO, AFC, FIFA WC   |   2    |Done|
| h_win_diff    |   1   |#Win T1 - T2         |   2    |Done|
| h_draw        |   1   |#Draw                |   2    |Done|
| rank_diff     |   1   |#Rank T1 - T2                 |   3    |Crawler|
| title_diff    |   1   |#Title won T1 - T2            |   3    |Crawler|
| f_goalF_1     |   2   |#Goal of T1 in 10 recent matches    |2|Done|
| f_goalF_2     |   2   |#Goal of T2 in 10 recent matches    |2|Done|
| f_goalA_1     |   2   |#Goal conceded of T1 in 10 recent matches    |2|Done|
| f_goalA_2     |   2   |#Goal conceded of T2 in 10 recent matches    |2|Done|
| f_win_1       |   2   |#Win of T1 in 10 recent matches     |2|Done|
| f_win_2       |   2   |#Win of T2 in 10 recent matches     |2|Done|
| f_draw_1      |   2   |#Draw of T1 in 10 recent matches     |2|Done|
| f_draw_2      |   2   |#Draw of T2 in 10 recent matches     |2|Done|
|ratio_odds     |   3   |average of bet odd t1 / t2           |4||
|avg_odds_draw  |   3   |average of bet odd of draw           |4||


### Train-Test
- Train set: International matches from 2000 - 2017 (exclude EURO 2016 and World Cup 2014)
- Test set : EURO 2016 and World Cup 2014

### Exploratory Data
Some image and observations. Some question for EDA

- Is playing as "Home Team" better than playing as "Away Team"?
![](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")



- Is there any difference between "form" of winning team and lose team
- How many time a bad-form team won a good-form team?
- What is a good-form / bad-form team?
- Is playing as "Home Team" better than playing as "Away Team"?
- Is ratio-odd usually right? How much are they likely to be correct?

# Model
Model I used and evaluation criteria
### Machine learning models
- Logistic Regression
- SVM with Gaussian Kernel
- Naive Bayes
- Decision Tree
- Random Forest
- Deep Learning

### Evaluation Criteria

# Results

# Reference
1. [A machine learning framework for sport result prediction](https://www.sciencedirect.com/science/article/pii/S2210832717301485)

# Task List
__Ongoing__
- [ ] Add feature group 1
    - [x] Add h_win_diff, h_draw
    - [ ] Add rank_diff, title_diff
- [ ] Add features group 3
- [ ] Add features group 4
- [ ] Build a data without player rating and squad value
- [ ] Build a web crawler for Ranking over time
- [ ] A table of title won for each team
- [ ] Integrate player rating and squad value to data
- [ ] Simple EDA and a small story

__Complete__
- [x] Add features group 2



