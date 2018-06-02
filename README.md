# Project Description
__Objective__:
- Prediction of the winner of an international matches Prediction results are  Win / Lose / Draw
- Apply the model to predict the result of FIFA world cup 2018.

__Data__: Data are assembled from multiple sources, most of them are from Kaggle, others come from FIFA website / EA games.

__Feature Engineering__: To determine who will more likely to win a match, based on my knowledge, I come up with 4 main groups of features as follows:
1. head-to-head match history
2. recent performance of each team (10 recent matches), aka "form"
3. bet-ratio before matches
4. squad value or how many players are in top 200 (from video game)
Feature list reflects those factors.

__Supervisor__: [Pratibha Rathore](https://www.linkedin.com/in/pratibha-rathore/)

__Lifecycle__

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/life_cycle.png)

__Report__:
Check the [Full Report](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/report/report.md) to gain more insight about this Project

# Data
### Data Source
The dataset are from all international matches from 2000 - 2017, results, bet odds, ranking, title won.
1. [FIFA World Cup 2018](https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data)
2. [International match 1872 - 2017](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data)
3. [Ranking through Time](http://www.fifa.com/fifa-world-ranking/associations/association=usa/men/index.html)
4. [Bet Odd](https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data)
5. [Player Rating Through time](https://www.futhead.com/10/players/?page=2)
6. [Squad for each tournament](https://github.com/openfootball/world-cup)
7. [FIFA player with time](https://www.fifaindex.com/players/fifa10_6/?nationality=1)

[1]: https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data
[2]: https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data
[3]: http://www.fifa.com/fifa-world-ranking/associations/association=usa/men/index.html
[4]: https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data
[5]: https://www.futhead.com/10/players/?page=2
[6]: https://github.com/openfootball/world-cup

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
| rank_diff     |   1   |#Rank T1 - T2                 |   3    ||
| title_diff    |   1   |#Title won T1 - T2            |   3    ||
| f_goalF_1     |   2   |#Goal of T1 in 10 recent matches    |2|Done|
| f_goalF_2     |   2   |#Goal of T2 in 10 recent matches    |2|Done|
| f_goalA_1     |   2   |#Goal conceded of T1 in 10 recent matches    |2|Done|
| f_goalA_2     |   2   |#Goal conceded of T2 in 10 recent matches    |2|Done|
| f_win_1       |   2   |#Win of T1 in 10 recent matches     |2|Done|
| f_win_2       |   2   |#Win of T2 in 10 recent matches     |2|Done|
| f_draw_1      |   2   |#Draw of T1 in 10 recent matches     |2|Done|
| f_draw_2      |   2   |#Draw of T2 in 10 recent matches     |2|Done|
|avg_odds_win_1 |   3   |average of bet odd for team 1        |4|Done|
|avg_odds_win_2 |   3   |average of bet odd for team 2        |4|Done|
|avg_odds_draw  |   3   |average of bet odd of draw           |4|Done|
|top_200        |   4   |number of players in top 200         |5||


# Preliminary Result

|           Model         |10-fold CV error rate (%)|
|:-----------------------:|:-------------------:|
|Logistic Regression      |59.05|
|SVM                      |59.17|
|Random Forest            |54.22|
|Gradient Boosting tree   |57.98|
|ADA boost tree           |59.01|
|Neural Net               |60.80|


### EURO 2016

We apply the model Gradient Boosted Tree to predict the result for EURO 2016 (not in the dataset)

|      team_1      |      team_2      | result | prediction |
|:----------------:|:----------------:|:------:|:----------:|
|      France      |      Romania     |   win  |     win    |
|      Albania     |    Switzerland   |  lose  |    lose    |
|      England     |      Russia      |  draw  |     win    |
|     Slovakia     |       Wales      |  lose  |    lose    |
|      Germany     |      Ukraine     |   win  |     win    |
| Northern Ireland |      Poland      |  lose  |    draw    |
|      Croatia     |      Turkey      |   win  |     win    |
|      Belgium     |       Italy      |  lose  |    lose    |
|      Ireland     |      Sweden      |  draw  |    lose    |
|  Czech Republic  |       Spain      |  lose  |    lose    |
|      Austria     |      Hungary     |  lose  |     win    |
|      Iceland     |     Portugal     |  draw  |    lose    |
|      Albania     |      France      |  lose  |    lose    |
|      Romania     |    Switzerland   |  draw  |     win    |
|      Russia      |     Slovakia     |  lose  |    lose    |
|      England     |       Wales      |   win  |     win    |
|      Germany     |      Poland      |  draw  |     win    |
| Northern Ireland |      Ukraine     |   win  |    draw    |
|      Croatia     |  Czech Republic  |  draw  |     win    |
|       Italy      |      Sweden      |   win  |     win    |
|       Spain      |      Turkey      |   win  |     win    |
|      Belgium     |      Ireland     |   win  |    draw    |
|      Hungary     |      Iceland     |  draw  |    lose    |
|      Austria     |     Portugal     |  draw  |    lose    |
|      Albania     |      Romania     |   win  |    lose    |
|      France      |    Switzerland   |  draw  |     win    |
|      Russia      |       Wales      |  lose  |    lose    |
|      England     |     Slovakia     |  draw  |     win    |
|      Croatia     |       Spain      |   win  |    lose    |
|  Czech Republic  |      Turkey      |  lose  |    lose    |
|      Germany     | Northern Ireland |   win  |     win    |
|      Poland      |      Ukraine     |   win  |     win    |
|      Hungary     |     Portugal     |  draw  |    lose    |
|      Austria     |      Iceland     |  lose  |     win    |
|      Ireland     |       Italy      |   win  |    lose    |
|      Belgium     |      Sweden      |   win  |    draw    |
|      Croatia     |     Portugal     |  lose  |    lose    |
|      Poland      |    Switzerland   |  draw  |    lose    |
| Northern Ireland |       Wales      |  lose  |    lose    |
|      France      |      Ireland     |   win  |     win    |
|      Germany     |     Slovakia     |   win  |     win    |
|      Belgium     |      Hungary     |   win  |     win    |
|      England     |      Iceland     |  lose  |     win    |
|       Italy      |       Spain      |   win  |    lose    |
|      Poland      |     Portugal     |  draw  |    lose    |
|      Belgium     |       Wales      |  lose  |    lose    |
|      Germany     |       Italy      |  draw  |     win    |
|      France      |      Iceland     |   win  |     win    |
|     Portugal     |       Wales      |   win  |     win    |
|      France      |      Germany     |   win  |    lose    |
|      France      |     Portugal     |  lose  |    draw    |

Accuracy = 0.48

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
- [ ] Add a simple classification based on "bet odd".
- [ ] Add graph of validation / training curve for hyper-parameters tuning.
- [ ] Add feature group 1
    - [x] Add h_win_diff, h_draw
    - [ ] Add rank_diff, title_diff
- [ ] Build a web crawler for Ranking over time
- [ ] A table of title won for each team
- [ ] Integrate player rating and squad value to data

__Complete__
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
