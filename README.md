# Project Description
__Objective__:
- Prediction of the winner of an international matches Prediction results are  "Win / Lose / Draw" or "goal difference"
- Apply the model to predict the result of FIFA world cup 2018.

__Data__: Data are assembled from multiple sources, most of them are from Kaggle, others come from FIFA website / EA games.

__Feature Engineering__: To determine who will more likely to win a match, based on my knowledge, I come up with 4 main groups of features as follows:
1. head-to-head match history between 2 teams
2. recent performance of each team (10 recent matches), aka "form"
3. bet-ratio before matches
4. squad strength (from FIFA video game)

Feature list reflects those factors.

__Supervisor__: [Pratibha Rathore](https://www.linkedin.com/in/pratibha-rathore/)

__Lifecycle__

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/life_cycle.png)

__Report__:
Check the [Full Report](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/report/report.md) to gain more insight about this Project

# Data
### Data Source
The dataset are from all international matches from 2000 - 2018, results, bet odds, ranking, squad strengths
1. [FIFA World Cup 2018](https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data)
2. [International match 1872 - 2018](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data)
3. [FIFA Ranking through Time](https://www.fifa.com/fifa-world-ranking/ranking-table/men/index.html)
4. [Bet Odd](https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data)
5. [Bet Odd 2](http://www.oddsportal.com)
6. [Squad Strength - Sofia](https://sofifa.com/players/top)
7. [Squad Strength - FIFA index](https://www.fifaindex.com/)



[1]: https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data
[2]: https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data
[3]: https://www.fifa.com/fifa-world-ranking/ranking-table/men/index.html
[4]: https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data
[5]: http://www.oddsportal.com
[6]: https://sofifa.com/players/top
[7]: https://www.fifaindex.com/

### Feature List
- _*difference: team1 - team2_
- _*form: performance in 10 recent matches_

| Feature Name  | Description              | Source |
|:-------------:|:------------------------:|:------:|
| team_1        | Nation Code (e.g US, NZ)      | [1] & [2] |
| team_2        | Nation Code  (e.g US, NZ)     | [1] & [2] |
| date          | Date of match yyyy - mm - dd  | [1] & [2] |
| tournament    | Friendly,EURO, AFC, FIFA WC   | [1] & [2] |
| h_win_diff    | Head2Head: win difference      |   [2]   |
| h_draw        | Head2Head: number of draw      |   [2]    |
| form_diff_goalF | Form: difference in "Goal For" |   [2]   |
| form_diff_goalA | Form: difference in "Goal Against" |   [2]    |
| form_diff_win   | Form: difference in number of win  |   [2]    |
| form_diff_draw  | Form: difference in number of draw |   [2]    |
| odd_diff_win    | Betting Odd: difference bet rate for win  | [4] & [5] |
| odd_draw        | Betting Odd: bet rate for draw            | [4] & [5] |
| game_diff_rank  | Squad Strength: difference in FIFA Rank   | [3] |
| game_diff_ovr   | Squad Strength: difference in Overall Strength  | [6] |
|game_diff_attk   | Squad Strength: difference in Attack Strength   | [6] |
|game_diff_mid    | Squad Strength: difference in Midfield Strength | [6] |
|game_diff_def    | Squad Strength: difference in Defense Strength  | [6] |
|game_diff_prestige | Squad Strength: difference in prestige        | [6] |
|game_diff_age11    | Squad Strength: difference in age of 11 starting players  | [6] |
|game_diff_ageAll   | Squad Strength: difference in age of all players          | [6] |
|game_diff_bup_speed| Squad Strength: difference in Build Up Play Speed         | [6] |
|game_diff_bup_pass | Squad Strength: difference in Build Up Play Passing       | [6] |
|game_diff_cc_pass  | Squad Strength: difference in Chance Creation Passing     | [6] |
|game_diff_cc_cross | Squad Strength: difference in Chance Creation Crossing    | [6] |
|game_diff_cc_shoot | Squad Strength: difference in Chance Creation Shooting    | [6] |
|game_diff_def_press| Squad Strength: difference in Defense Pressure            | [6] |
|game_diff_def_aggr | Squad Strength: difference in Defense Aggression          | [6] |
|game_diff_def_teamwidth  | Squad Strength: difference in Defense Team Width    | [6] |

## Feature Importance:
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/feature_importance.png)

From the graph, we can see that among 4 feature groups "head2head","form", "bet odd" and "squad strength", "bet odd" group plays more important role to predict the match results than other feature groups. In "squad strength" feature, "defense aggression", "build up play passing" and "chance creation Shooting" help us determine goal difference better than other features.

# Preliminary Result

|           Model         |10-fold CV error rate (%)|
|:-----------------------:|:-------------------:|
|Logistic Regression      |59.05|
|SVM                      |59.17|
|Random Forest            |54.22|
|Gradient Boosting tree   |57.98|
|ADA boost tree           |59.01|
|Neural Net               |60.80|

### World Cup 2018
Now the model is applying for World Cup 2018 in Russia with __simulation time = 100 000__. While the model didn't perform so well in train/test phase, it has shown stunning results so far, 8 correct results out of 14 matches including matches such as __Portugal - Spain (3-3)__, __Brazil - Switzerland (1-1)__. Here is the result of my model for all matches in Match Day 1 of Group Stage

__Result Explanation:__

Team A vs Team B

- "win_1": A wins with 1 goal differences
- "win_2": A wins with 2 goal differences
- "win_3": A wins with 3 or more goal differences
- "lose_1": B wins with 1 goal differences
- "lose_2": B wins with 2 goal differences
- "lose_3": A wins with 3 or more goal differences
- "draw_0": Draw

__Match Day 1__
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/WC_2018_matchday1.PNG)

Accuracy = 8 / 15

__Match Day 2__ (coming soon)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/WC_2018_matchday2.PNG)

Accuracy = 

__Match Day 3__ (coming soon)

# Reference
1. [A machine learning framework for sport result prediction](https://www.sciencedirect.com/science/article/pii/S2210832717301485)
2. [t-test definition](https://en.wikipedia.org/wiki/Student%27s_t-test)
3. [Confusion Matrix Multi-Label example](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
4. [Precision-Recall Multi-Label example](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#in-multi-label-settings)
5. [ROC curve example](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)
6. [Model evaluation](http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
7. [Tuning the hyper-parameters of an estimator](http://scikit-learn.org/stable/modules/grid_search.html)
8. [Validation curves](http://scikit-learn.org/stable/modules/learning_curve.html)
9. [Understand Bet odd format](https://www.pinnacle.com/en/betting-articles/educational/odds-formats-available-at-pinnacle-sports/ZWSJD9PPX69V3YXZ)
10. [EURO 2016 bet odd](http://www.oddsportal.com/soccer/europe/euro-2016/results/#/)

# Task List
__Ongoing__
- [ ] Add prediction for Matchday 3
- [ ] Add summary result for Matchday 1
- [ ] Add graph of validation / training curve for hyper-parameters tuning.

__Complete__
- [x] Add prediction for Matchday 2
- [x] Add feature Importance
- [x] Add feature of squad and player info
- [x] Build a web crawler for Squad each team
- [x] Build a web crawler for FIFA game player
- [x] Add a simple classification based on "bet odd".
- [x] Add feature group 1
    - [x] Add h_win_diff, h_draw
    - [x] Add rank_diff, title_diff
- [x] Add features group 2
- [x] Add features group 3
- [x] Simple EDA and a small story
- [x] Add features group 4
- [x] Prepare framework for running classifiers
- [x] Add evaluation metrics and plot
  - [x] Add accuracy, precision, recall, F1
  - [x] Add ROC curves
- [x] Build a data without player rating and squad value
- [x] Generate data and preform prediction for EURO 2016, ok now my story is more interesting
- [x] Create more data, "teamA vs teamB -> win" is equivalent to "teamB vs teamA -> lose"
