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

[Source 4]: https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data

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
There are few questions in order to understand data better

1. Is playing as "Home Team" better than playing as "Away Team"?

   To answer this question, we investigate the how frequent a home-team wins a match.
   According to the bar graph, home teams are more likely, twice exactly, to win the game.

    ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/home_team.png)

2. Does head-to-head matchup history affect the current match?

    Now we compare the difference between "win difference" of winner / loser and draw-er
    "Win difference" is defined as number of wins of team A - number of wins of team B
    ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/win_diff.png)

    Perform to t-test with hypothesis "mean of win difference between winner and loser are the same"
    ```python
    Ttest_indResult(statistic=35.432781367462361, pvalue=5.3865361955241691e-266)
    ```
    Very small of p-value means we can reject the hypothesis. Therefore, we can say __history of head-to-head matches of two
    teams contribute significantly to the result__

3. Is there any difference between "form" of winning team and lose team?
    We compare the difference between goals / goals conceded / number of wins / number of draws of winner, loser or draw-er.

    ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/recent_form.png)

    As you can see, while goals, goals conceded and number of draw show no significant difference, number of wins tell us more about who is more likely to win.
    t-test also confirms our assumption
    ```python
    Ttest_indResult(statistic=29.488698758378064, pvalue=9.6646508941629036e-187)
    ```
- How many time a bad-form team won a good-form team?
- What is a good-form / bad-form team?

4. Is ratio-odd usually right? How much are they likely to be correct?
    For this question, we use the average odd getting from [Bet Odd][Source 4] before matches.

    ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/avg_odd_win.png)


    ```python
    Ttest_indResult(statistic=-43.22461132125629, pvalue=0.0)
    ```

    We can say that we can reply on bet odd to predict the match results.

# Model
Model we used and evaluation criteria
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

__Complete__
- [x] Add features group 2
- [x] Simple EDA and a small story


