
# Table of Contents
1. [Introduction](#introduction)
2. [Data](#data): How do you determine feature list and how do you collect data?
3. [Exploratory Data Analysis](#exploratory-data-analysis): charts and hypothesis testing
    1. [Correlation between variables]()
    2. [How head-to-head matchup history affect the current match?]()
    3. [How recent performances affect the current match?]()
    4. [Is ratio-odd usually right? How much are they likely to be correct?]()
    5. [Do strong teams usually win?]()
    6. [Data Distrubution in PCA]()
4. [Methodology](#methodology): Details about your procedure
    1. [Classifiers](#classifiers): Definition and parameters meaning
        - [Dummy Classifiers]() Define a dummy classifier
        - [Logistic Regression]()
        - [Support Vector Machine]()
        - [Ensemble Trees]()
        - [Neural Network]()
    2. [Evaluation Criteria](#evaluation-criteria): Definition, strength and weakness
        - [Accuracy]()
        - [Recall, Precision, F1]()
        - [Out of Bag Error]()
        - [10-fold cross validation error]()
    3. [Hyper Parameter Tuning](#hyper-parameter-yuning)

4. [Results](#results)
5. [Data Source](#data-source)
6. [References](#references)
7. [Appendix](#appendix)

# Introduction
__Objective__:
- Prediction of the winner of an international matches Prediction results are  "Win / Lose / Draw" or "goal difference"
- Apply the model to predict the result of FIFA world cup 2018.

__Supervisor__: [Pratibha Rathore](https://www.linkedin.com/in/pratibha-rathore/)

__Lifecycle__

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/life_cycle.png)

__Previous Work__:


# Data
__Data__: The dataset are from all international matches from 2000 - 2018, results, bet odds, ranking, squad strengths.
1. [FIFA World Cup 2018][1]
2. [International match 1872 - 2018][2]
3. [FIFA Ranking through Time][3]
4. [Bet Odd][4]
5. [Bet Odd 2][5]
6. [Squad Strength - Sofia][6]
7. [Squad Strength - FIFA index][7]

[1]: https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data
[2]: https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data
[3]: https://www.fifa.com/fifa-world-ranking/ranking-table/men/index.html
[4]: https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data
[5]: http://www.oddsportal.com
[6]: https://sofifa.com/players/top
[7]: https://www.fifaindex.com/

__Feature Selection__: To determine who will more likely to win a match, based on my knowledge, I come up with 4 main groups of features as follows:
1. __head-to-head match history between 2 teams__. Some teams have few opponents who they hardly win no matter how strong they currently are. For example [Germany team usually loses / couldn't beat Italian team in 90 minute matches.](https://www.11v11.com/teams/italy/tab/opposingTeams/opposition/Germany/)
2. __Recent performance of each team (10 recent matches), aka "form"__ A team with "good" form usually has higher chance to win next matches.
3. __Bet-ratio before matches__ Odd bookmarkers already did many analysis before matches to select the best betting odds so why don't we include them.
4. __Squad strength (from FIFA video game).__ We want a real squad strength but these data are not free and not always available so we use strength from FIFA video games which have updated regularly to catch up with the real strength.

__Feature List__ Feature list reflects those four factors.

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


# Exploratory Data Analysis
There are few questions in order to understand data better

__1. Correlation between variables__

First we draw correlation matrix of large dataset which contains all matches from 2005-2018 with features group 1,2 and 3
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/corr_matrix.png)

In general, features are not correlated. "odd_win_diff" is quite negatively correlated with "form_diff_win" (-0.5), indicating that form of two teams reflex belief of odd bookmarkers on winners. One more interesting point is when difference of bet odd increases we would see more goal differences (correlation score = -0.6).
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/odd-vs-goal.png)

Second, we draw correlation matrix of small dataset which contains all matches from World Cup 2010, 2014, 2018 and EURO 2012, 2016
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/corr_matrix_full.png)

Overall rating is just an average of "attack", "defense" and "midfield" index therefore we see high correlation between them. In addition, some of new features of squad strength show high correlation for example "FIFA Rank", "Overall rating" and "Difference in winning odd"
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/rank-odd-rating.png)

__2. How head-to-head matchup history affect the current match?__

You may think when head-to-head win difference is positive, match result should be "Win" (Team 1 wins Team 2) and vice versa, when head-to-head win difference is negative, match result should be "Lose" (Team 2 wins Team 1). In fact, positive head-to-head win difference indicates that there is 51.8% chance the match results end up with "Win" and negative head-to-head win difference indicates that there is 55.5% chance the match results end up with "Lose"
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-positive-h2h.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-negative-h2h.png)

Let's perform our hypothesis testing with two-sampled t-test
Null Hypothesis: There is no difference of 'h2h win difference' between "Win" and "Lose"
Alternative Hypothesis: There are differences of 'h2h win difference' between "Win" and "Lose"

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/win_diff.png)

```
T-test between win and lose:
Ttest_indResult(statistic=24.30496036405259, pvalue=2.503882847793891e-126)
```
Very small of p-value means we can reject the null hypothesis and accept alternative hypothesis.

We can do the same procedure with win-draw and lose-draw

```
T-test between win and draw:
Ttest_indResult(statistic=7.8385466293651023, pvalue=5.395456011352264e-15)

T-test between lose and draw:
Ttest_indResult(statistic=-8.6759649601068887, pvalue=5.2722587025773183e-18)

```
Therefore, we can say __history of head-to-head matches of two teams contribute significantly to the result__

__3. How 10-recent performances affect the current match?__

We consider differences in "Goal For" (how many goals they got), "Goal Against" (how many goals they conceded), "number of winning matches" and "number of drawing matches". We performed same procedure as previous questions. From pie charts, we can see a clear distinction in "number of wins" where proportion of "Win" result decreases from 49% to 25% while "Lose" result increases from 26.5% to 52.3%.

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-form-diff-goalF.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-form-diff-goalA.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-form-diff-win.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-form-diff-draw.png)

Pie charts are not enough we should do the hypothesis testing to see significance of each feature

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| Goal For | pvalue = 2.50e-126 | pvalue = 5.39e-15 | pvalue = 5.27e-18 |
| Goal Against | pvalue = 0.60 | pvalue = 0.17 | pvalue = 0.08 |
| Number of Winning Matches | pvalue = 3.02e-23 | pvalue = 1.58e-33 | pvalue = 2.57e-29 |
| Number of Draw Matches | pvalue = 1.53e-06 | pvalue = 0.21 | pvalue = 0.03 |

We see many small value of p-value in cases of "Goal For" and "Number of Winning Matches". Based on t-test, __we know difference in "Goal For" and "Number of Winning Matches" are helpful features__

__4. Do stronger teams usually win?__

We define stronger teams based on
 - Higher FIFA Ranking
 - Higher Overall Rating


![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-game-diff-rank.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-game-rating.png)

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| FIFA Rank | pvalue = 2.11e-10 | pvalue=0.65 | pvalue=0.00068 |
| Overall Rating| pvalue = 1.53e-16 | pvalue = 0.0804 | pvalue = 0.000696 |



__5. Do young players play better than old one ?__

Young players may have better stamina and more energy while older players have more experience. We want to see how age affects match results.

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-game-diff-age11.png)

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| Age | pvalue = 2.07e-05| pvalue = 0.312 | pvalue=0.090 |

Based on t-test and pie chart, we know that the age contributes significantly to the result. More specifically, younger teams tends to play better than older ones

__6. Is short pass better than long pass ?__
Higher value of "Build Up Play Passing" means "Long Pass" and lower value  means "Short Pass", value in middle mean "Mixed-Type Pass"

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/pie-game-diff-bup-pass.png)

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| Age | pvalue = 1.05e-07| pvalue = 0.0062 | pvalue = 0.571 |

Based on t-test and pie chart, we know that the age contributes significantly to the result. More specifically, teams who replies on "Longer Pass" usually loses the game.

__7. How does crossing pass affect match result ?__
__8. How does chance creation shooting affect match result ?__
__9. How does defence pressure affect match result ?__
__10. How does defence aggression affect match result ?__
__11. How does defence team width affect match result ?__


4. Is ratio-odd usually right? How much are they likely to be correct?
    For this question, we use the average odd getting from [Bet Odd][2] before matches.

    ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/avg_odd_win.png)


    ```python
    Ttest_indResult(statistic=-43.22461132125629, pvalue=0.0)
    ```

    We can say that we can reply on bet odd to predict the match results.
5. How labels distribute in reduced dimension?

  For this question, we use PCA to pick two first principal components which best explained data. Then we plot data in new dimension

   ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/2pc.png)

   While "Win" and "Lose" are while separate, "Draw" seems to be mixed between other labels.

# Methodology
Our main objectives of prediction are "Win / Lose / Draw" and "Goal Difference".
In this work, we do two main experiments:

 1. Build classifiers for "Win / Lose / Draw" from 2005. Because feature "Bet Odds" are only available after 2005 so we only conduct experiments for this period of time.

 2. Build classifiers for "Goal Difference" for "World Cup" and "UEFA EURO" after 2010. The reason is because features of "Squad Strength" are not always available before 2010, some national teams does not have database of squad strength in FIFA Video Games. We know that tackling prediction with regression would be hard so we turn "Goal Difference" into classification by defining labels as follows:

 __Team A vs Team B__
 - "win_1": A wins with 1 goal differences
 - "win_2": A wins with 2 goal differences
 - "win_3": A wins with 3 or more goal differences
 - "lose_1": B wins with 1 goal differences
 - "lose_2": B wins with 2 goal differences
 - "lose_3": A wins with 3 or more goal differences
 - "draw_0": Draw

For each experiment we follow these procedure
 - Split data into 70:30
 - First we perform "normalization" of features, convert category to number
 - Second we perform k-fold cross validation to select the best parameters for each model based on some criteria.
 - Third we use the best model to do prediction on 10-fold cross validation (9 folds for training and 1 fold for testing) to achieve the mean of test error. This error is more reliable.

## Models
__Baseline Model:__

Bet odds are results of analysis from bet bookmarkers to each match, therefore we want to see whether we can beat the bet odds to determine who is more likely to win. However, we also want to predict "Draw" matches so we use the following rules for Win / Lose / Draw as a baseline
---- Rules -----

We also take advantage of "dummy classifier" from sklearn library which automatically derive a simple rule for classification

__Enhanced Model:__

To beat the baseline model we try to use several machine algorithm as follows
1. Logistic Regression
2. SVM
3. Random Random Forest
4. Gradient Boosting Tree
5. ADA Boost Tree
6. Neural Network


## Evaluation Criteria
Models are evaluated on these criteria which are carried out for each label "win", "lose" and "draw"
- __Precision__: Among our prediction of "True" value, how many percentage we hit?, the higher value, the better prediction
- __Recall__: Among actual "True" value, how many percentage we hit?, the higher value, the better prediction
- __F1__: A balance of Precision and Recall, the higher value, the better prediction, there are 2 types of F1
  - __F1-micro__:
  - __F1-macro__:
- __10-fold cross validation test error__: A reliable estimation of test error of model evaluation (no need to split to train and test)
- __ROC curve__:

# Results

|           Model         |10-fold CV error rate (%)|
|:-----------------------:|:-------------------:|
|Logistic Regression      |59.05|
|SVM                      |59.17|
|Random Forest            |54.22|
|Gradient Boosting tree   |57.98|
|ADA boost tree           |59.01|
|Neural Net               |60.80|

1. __Logistic Regression__

  Best parameters:
  ```python
  LogisticRegression(C=0.00046415888336127773, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='multinomial', n_jobs=1, penalty='l2',
          random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
          warm_start=False)
  ```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_lr]    | ![alt text][roc_lr]  |

[cm_lr]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/cm_lr.png
[roc_lr]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/roc_lr.png

| Label | Precision | Recall | F1-score |
|:-----:|:---------:|:------:|:--------:|
| Draw  |   0.25    |   0.01 |   0.01   |
| Win   |   0.60    |   0.85 |   0.70   |
| Lose  |   0.61    |   0.74 |   0.66   |
| avg / total |   0.51  |    0.60   |   0.52   |

10-fold CV Test accuracy = 0.5905079026832715

2. __SVM__

Best parameters:
```python
grid_SVM = [{'kernel': ['rbf','linear'], 'C': np.logspace(-6, 0, 3)}]
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_SVM]    | ![alt text][roc_SVM]  |

[cm_SVM]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/cm_svm.png
[roc_SVM]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/roc_svm.png

| Label | Precision | Recall | F1-score |
|:-----:|:---------:|:------:|:--------:|
| Draw  |   0.00    |   0.00 |   0.00   |
| Win   |   0.60    |  0.82  |    0.69   |
| Lose  |    0.59   |   0.75 |     0.66 |
| avg / total |   0.44  |    0.59  |    0.51    |

Test Accuracy = 0.591753774680604

3. __Random Forest__

Current parameters:
```python
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2000, n_jobs=-1,
            oob_score=False, random_state=0, verbose=True,
            warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_rf]    | N/A  |

[cm_rf]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/cm_rf.png


| Label | Precision | Recall | F1-score |
|:-----:|:---------:|:------:|:--------:|
| Draw  |   0.37    |  0.08  |    0.14   |
| Win   |   0.60    |  0.80  |    0.68  |
| Lose  |   0.61    |  0.71  |    0.66    |
| avg / total |    0.55  |    0.59  |    0.54    |

10-fold CV Test accuracy = 0.5422216762135833

4. __Gradient Boosting tree__

Best parameters:
```python
grid_GBT = [{'max_depth': [3,5,7], 'n_estimators': [100,1000,2000]}]
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              presort='auto', random_state=0, subsample=1.0, verbose=True,
              warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_gbt]    | ![alt text][roc_gbt]  |

[cm_gbt]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/cm_gbt.png
[roc_gbt]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/roc_gbt.png

| Label | Precision | Recall | F1-score |
|:-----:|:---------:|:------:|:--------:|
| Draw  |   0.39    |  0.11  |    0.18    |
| Win   |    0.61   |   0.80 |     0.69   |
| Lose  |  0.61     | 0.71   |   0.66    |
| avg / total |   0.56   |   0.60  |    0.55   |

10-fold CV Test accuracy = 0.5798773361822063

5. __ADA boost tree__

Best parameters:
```python
grid_ADA = [{'base_estimator': [DT_3,DT_5], 'n_estimators': [100,1000,2000,3000]}]
AdaBoostClassifier(algorithm='SAMME',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=100, random_state=None)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_ada]    | ![alt text][roc_ada]    |

[cm_ada]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/cm_ada.png
[roc_ada]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/roc_ada.png

| Label | Precision | Recall | F1-score |
|:-----:|:---------:|:------:|:--------:|
| Draw  |  0.32     | 0.15   |   0.20   |
| Win   |   0.60    |  0.75  |    0.67  |
| Lose  |   0.60    |  0.66  |    0.63    |
| avg / total |   0.53  |    0.57  |    0.54    |

10-fold CV Test accuracy = 0.5901621943365096

6. __Neural Net__

Best parameters:
```python
MLPClassifier(hidden_layer_sizes = (13,10), max_iter = 1000, alpha=1e-4,
                    solver='adam', verbose=True, tol=1e-10, random_state=1, learning_rate_init=.1)
```
| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_nn]    | N/A  |

[cm_nn]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/cm_nn.png


| Label | Precision | Recall | F1-score |
|:-----:|:---------:|:------:|:--------:|
| Draw  |   0.00    |   0.00 |   0.00   |
| Win   |   0.62    |  0.82  |    0.71  |
| Lose  |   0.59    |  0.80  |    0.68   |
| avg / total |   0.46   |   0.61  |    0.52  |


10-fold CV Test accuracy = 0.60801393728223


# Data Source
The dataset are from all international matches from 2000 - 2018, results, bet odds, ranking, squad strengths
1. [FIFA World Cup 2018](https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data)
2. [International match 1872 - 2018](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data)
3. [FIFA Ranking through Time](https://www.fifa.com/fifa-world-ranking/ranking-table/men/index.html)
4. [Bet Odd](https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data)
5. [Bet Odd 2](http://www.oddsportal.com)
6. [Squad Strength - Sofia](https://sofifa.com/players/top)
7. [Squad Strength - FIFA index](https://www.fifaindex.com/)


[1]: https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data
[2]: https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data
[3]: http://www.fifa.com/fifa-world-ranking/associations/association=usa/men/index.html
[4]: https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data
[5]: https://www.futhead.com/10/players/?page=2
[6]: https://github.com/openfootball/world-cup

# References
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

# Appendix
Challenge / Question
1. If "teamA vs teamB -> win" is equivalent to "teamB vs teamA -> lose", will adding these data make model better? (You actually did it in EDA)
2. According to "draw" and "win" in average odd? These two labels seem to be different, why performance of "draw" is bad?
3. How about classification -> regression
