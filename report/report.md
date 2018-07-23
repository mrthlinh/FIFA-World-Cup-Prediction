
# Table of Contents
1. [Introduction](#introduction)
2. [Data](#data):
3. [Exploratory Data Analysis](#exploratory-data-analysis):
    1. Correlation between variables
    2. How head-to-head matchup history affect the current match?
    3. How recent performances affect the current match?
    4. Do strong teams usually win?
    5. Do young players play better than old one ?
    6. Is short pass better than long pass ?
    7. How labels distribute in reduced dimension?
4. [Methodology](#methodology): Details about your procedure
5. [Models](#models)
    1. Baseline models
      - odd-based model
      - history and form based model
      - squad-strength-based model
    2. Enhance models
      - Logistic Regression
      - Random Forest
      - Gradient Boosting tree
      - ADA boost tree
      - Neural Network
      - Light GBM
6. [Evaluation Criteria](#evaluation-criteria)
  - F1
  - 10-fold Cross Validation accuracy
  - Area under ROC
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [References](#references)
10. [Appendix](#appendix)

# Introduction

__Abstract__:

In this work, we compare 9 different modeling approaches for the soccer matches and goal difference on all international matches from 2005 - 2017,  FIFA World Cup 2010 - 2014 and FIFA EURO 2012-2016. Within this comparison, while performance of "Win / Draw / Lose" predictions shows not much difference, "Goal Difference" prediction is quite favored to Random Forest and squad-strength based decision tree. We also apply these models in World Cup 2018 and again, Random Forest and Logistic Regression predicts about 33% acccuracy for "Goal Difference" and about 57% for "Win / Draw / Lose". However a simple decision tree based on bet odd and squad-strength are also comparable.

__Objective__:
- Prediction of the winner of an international matches Prediction results are  "Win / Lose / Draw" or "goal difference"
- Apply the model to predict the result of FIFA world cup 2018.

__Supervisor__: [Pratibha Rathore](https://www.linkedin.com/in/pratibha-rathore/)

__Lifecycle__

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/life_cycle.png)

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

__Imbalance of data__

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/class_imbalance_1.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/class_imbalance_2.png)

__Correlation between variables__

First we draw correlation matrix of large dataset which contains all matches from 2005-2018 with features group 1,2 and 3
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/corr_matrix.png)

In general, features are not correlated. "odd_win_diff" is quite negatively correlated with "form_diff_win" (-0.5), indicating that form of two teams reflex belief of odd bookmarkers on winners. One more interesting point is when difference of bet odd increases we would see more goal differences (correlation score = -0.6).
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/odd-vs-goal.png)

Second, we draw correlation matrix of small dataset which contains all matches from World Cup 2010, 2014, 2018 and EURO 2012, 2016
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/corr_matrix_full.png)

Overall rating is just an average of "attack", "defense" and "midfield" index therefore we see high correlation between them. In addition, some of new features of squad strength show high correlation for example "FIFA Rank", "Overall rating" and "Difference in winning odd"
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/rank-odd-rating.png)

__How head-to-head matchup history affect the current match?__

You may think when head-to-head win difference is positive, match result should be "Win" (Team 1 wins Team 2) and vice versa, when head-to-head win difference is negative, match result should be "Lose" (Team 2 wins Team 1). In fact, positive head-to-head win difference indicates that there is 51.8% chance the match results end up with "Win" and negative head-to-head win difference indicates that there is 55.5% chance the match results end up with "Lose"
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-positive-h2h.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-negative-h2h.png)

Let's perform our hypothesis testing with two-sampled t-test
Null Hypothesis: There is no difference of 'h2h win difference' between "Win" and "Lose"
Alternative Hypothesis: There are differences of 'h2h win difference' between "Win" and "Lose"

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/win_diff.png)

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

__How 10-recent performances affect the current match?__

We consider differences in "Goal For" (how many goals they got), "Goal Against" (how many goals they conceded), "number of winning matches" and "number of drawing matches". We performed same procedure as previous questions. From pie charts, we can see a clear distinction in "number of wins" where proportion of "Win" result decreases from 49% to 25% while "Lose" result increases from 26.5% to 52.3%.

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-form-diff-goalF.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-form-diff-goalA.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-form-diff-win.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-form-diff-draw.png)

Pie charts are not enough we should do the hypothesis testing to see significance of each feature

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| Goal For | pvalue = 2.50e-126 | pvalue = 5.39e-15 | pvalue = 5.27e-18 |
| Goal Against | pvalue = 0.60 | pvalue = 0.17 | pvalue = 0.08 |
| Number of Winning Matches | pvalue = 3.02e-23 | pvalue = 1.58e-33 | pvalue = 2.57e-29 |
| Number of Draw Matches | pvalue = 1.53e-06 | pvalue = 0.21 | pvalue = 0.03 |

We see many small value of p-value in cases of "Goal For" and "Number of Winning Matches". Based on t-test, __we know difference in "Goal For" and "Number of Winning Matches" are helpful features__

__Do stronger teams usually win?__

We define stronger teams based on
 - Higher FIFA Ranking
 - Higher Overall Rating


![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-game-diff-rank.png)
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-game-rating.png)

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| FIFA Rank | pvalue = 2.11e-10 | pvalue=0.65 | pvalue=0.00068 |
| Overall Rating| pvalue = 1.53e-16 | pvalue = 0.0804 | pvalue = 0.000696 |



__Do young players play better than old one ?__

Young players may have better stamina and more energy while older players have more experience. We want to see how age affects match results.

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-game-diff-age11.png)

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| Age | pvalue = 2.07e-05| pvalue = 0.312 | pvalue=0.090 |

Based on t-test and pie chart, we know that the age contributes significantly to the result. More specifically, younger teams tends to play better than older ones

__Is short pass better than long pass ?__
Higher value of "Build Up Play Passing" means "Long Pass" and lower value  means "Short Pass", value in middle mean "Mixed-Type Pass"

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/pie-game-diff-bup-pass.png)

| Feature Name  | t-test between 'win' and 'lose' | t-test between 'win' and 'draw' | t-test between 'lose' and 'draw' |
|:-------------:|:-------------------------------:|:-------------------------------:|:--------------------------------:|
| Age | pvalue = 1.05e-07| pvalue = 0.0062 | pvalue = 0.571 |

Based on t-test and pie chart, we know that the age contributes significantly to the result. More specifically, teams who replies on "Longer Pass" usually loses the game.

__How does crossing pass affect match result ?__

__How does chance creation shooting affect match result ?__

__How does defence pressure affect match result ?__

__How does defence aggression affect match result ?__

__How does defence team width affect match result ?__

__How labels distribute in reduced dimension?__

  For this question, we use PCA to pick two first principal components which best explained data. Then we plot data in new dimension

   ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/EDA/2pc.png)

   While "Win" and "Lose" are while separate, "Draw" seems to be mixed between other labels.

# Methodology
Our main objectives of prediction are "Win / Lose / Draw" and "Goal Difference".
In this work, we do two main experiments, for each experiment we follow these procedure
 - Split data into 70:30
 - First we perform "normalization" of features, convert category to number
 - Second we perform k-fold cross validation to select the best parameters for each model based on some criteria.
 - Third we use the best model to do prediction on 10-fold cross validation (9 folds for training and 1 fold for testing) to achieve the mean of test error. This error is more reliable.

__Experiment 1.__ Build classifiers for "Win / Lose / Draw" from 2005. Because feature "Bet Odds" are only available after 2005 so we only conduct experiments for this period of time.

__Experiment 2.__ Build classifiers for "Goal Difference" for "World Cup" and "UEFA EURO" after 2010. The reason is because features of "Squad Strength" are not always available before 2010, some national teams does not have database of squad strength in FIFA Video Games. We know that tackling prediction with regression would be hard so we turn "Goal Difference" into classification by defining labels as follows:

 __Team A vs Team B__
 - "win_1": A wins with 1 goal differences
 - "win_2": A wins with 2 goal differences
 - "win_3": A wins with 3 or more goal differences
 - "lose_1": B wins with 1 goal differences
 - "lose_2": B wins with 2 goal differences
 - "lose_3": A wins with 3 or more goal differences
 - "draw_0": Draw

__Experiment 3.__ In addition, we want to test how our trained model in __Experiment 2__ to predict the "Goal Difference" and "Win/Draw/Lose" of matches in World Cup 2018.

## Models
__Baseline Model:__
In EDA part, we already investigate importance of features and see that odd, history, form and squad strength are all significant. Now we divide features into three groups: odd, h2h-form, squad strength and build "Baseline Models" based on these groups. To keep the baseline model simple, we set hyper-parameter of Decision Tree maximum depth = 2, maximum leaf nodes = 3

1. __Odd-based model:__

| Experiment 1 | Experiment 2 |
|:------------:|:------------:|
| ![alt text][tree_odd_1] | ![alt text][tree_odd_2]  |

[tree_odd_1]:https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/ex1/tree-odd.png
[tree_odd_2]:https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/ex2/tree-odd.png

2. __History-Form-based model:__

| Experiment 1 | Experiment 2 |
|:------------:|:------------:|
| ![alt text][tree_odd_1] | ![alt text][tree_odd_2]  |

[tree_h2h_1]:https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/ex1/tree-h2h-form.png
[tree_h2hd_2]:https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/ex2/tree-h2h-form.png


3. __Squad-strength based model:__

For experiment 2

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/ex2/tree-ss.png)

__Enhanced Model:__

To beat the baseline models we use all features and several machine algorithms as follows

1. Logistic Regression
2. Random Forest
3. Gradient Boosting Tree
4. ADA Boost Tree
5. Neural Network
6. LightGBM


## Evaluation Criteria
Models are evaluated on these criteria which are carried out for each label "win", "lose" and "draw"
- __Precision__: Among our prediction of "True" value, how many percentage we hit?, the higher value, the better prediction

- __Recall__: Among actual "True" value, how many percentage we hit?, the higher value, the better prediction

- __F1__: A balance of Precision and Recall, the higher value, the better prediction, there are 2 types of F1
  - __F1-micro__: compute F1 by aggregating True Positive and False Positive or each class
  - __F1-macro__: compute F1 independently for each class and take the average (all classed equally)

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/micro-vs-macro-f1.PNG)

In a multi-class classification setup, micro-average is preferable if you suspect there might be class imbalance (i.e you may have many more examples of one class than of other classes). In this case, we should stick with F1-micro

- __10-fold cross validation accuracy__: Mean of accuracy for each cross-validation fold. This is a reliable estimation of test error of model evaluation (no need to split to train and test)

- __Area under ROC__: For binary classification, True Positive Rate vs False Positive Rate for all threshold.
![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/roc_intro.png)

# Results

__Experiment 1__ "Draw / Lose /Win"

|           Model         |10-fold CV accuracy (%)| F1 - micro average | AUROC - micro average |
|:-----------------------:|:---------------------:|:------------------:|:---------------------:|
|Odd-based Decision Tree  |59.28|60.22|0.76|
|H2H-Form based Decision Tree  |51.22|51.52|0.66|
|Logistic Regression      |59.37|59.87|0.76|
|Random Forest            |54.40|55.92|0.74|
|Gradient Boosting tree   |58.60|59.47|0.77|
|ADA boost tree           |59.08|60.22|0.77|
|Neural Net               |58.96|58.36|0.77|
|LightGBM                 |59.49|60.28|0.78|

Results from experiment 1 show little improvement between enhanced models and baseline models based on three evaluation criteria: 10-fold cross validation, F1 and Area Under Curve. A simple Odd-based Decision Tree is enough to classify Win/Draw/Lose . However, according to confusion matrix in [Appendix](#appendix) of experiment 1, we see that most of classifiers failed to classify "Draw" label, only Random Forest and Gradient Boosting Tree can predict "Draw" label, 74 hits and 29 hits respectively. Furthermore, as we mentioned, there is not much difference of classifiers in other criteria so our recommendation for classify "Win / Draw / Lose" is __"Gradient Boosting Tree"__ and __"Random Forest"__

__Experiment 2__ "Goal Difference"

|           Model         |10-fold CV accuracy (%)| F1 - micro average | AUROC - micro average |
|:-----------------------:|:---------------------:|:------------------:|:---------------------:|
|Odd-based Decision Tree  |26.41|25.37|0.62|
|H2H-Form-based Decision Tree  |16.74|18.94|0.59|
|Squad-strength-based Decision Tree  |31.64|31.34|0.66|
|Logistic Regression      |21.39|22.38|0.64|
|Random Forest            |25.36|25.37|0.60|
|Gradient Boosting tree   |27.27|16.42|0.58|
|ADA boost tree           |26.92|16.41|0.59|
|Neural Net               |22.42|25.37|0.63|
|LightGBM                 |25.62|20.89|0.57|

In experiment 2, "Squad Strength" based Decision Tree tends to superior to other classifiers.

__Experiment 3__ "Goal Difference" and "Win/Draw/Lose" in World Cup 2018

|           Model         |"Goal Difference" Accuracy| "Win/Draw/Lose" Accuracy (%)| F1 - micro average |
|:-----------------------:|:------------------------:|:---------------------------:|:------------------:|
|Odd-based Decision Tree              |31.25|48.43|31.25|
|H2H-Form based Decision Tree         |25.00|34.37|25.00|
|Squad strength based Decision Tree   |28.12|43.75|28.12|
|Logistic Regression                  |32.81|57.81|32.81|
|Random Forest                        |32.81|56.25|32.81|
|Gradient Boosting tree               |21.87|45.31|21.87|
|ADA boost tree                       |28.12|51.56|28.12|
|Neural Net                           |20.31|35.94|20.31|
|LightGBM                             |32.81|56.25|32.81|

# Conclusion

In conclusion, odd-based features from bet bookmarkers are reliable to determine who is the winner of matches. However, it is very bad at finding out whether matches end up a draw result. Instead, Ensemble method like Random Forest and Gradient Boosting tree are superior in this case. Squad index from FIFA video games provide more information and also contribute significantly for prediction of "Goal Difference". Other complex machine learning models show not much difference against simple odd-based or strength-based tree, this is reasonable because the amount of data are limited and a simple decision tree can provide an easy solution.


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

__Experiment 1__

1. __Odd-based Decision Tree__:

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_odd]    | ![alt text][roc_odd]  |

[cm_odd]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-odd.png
[roc_odd]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/roc-odd.png

2. __h2h-Form-based Decision Tree__:

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_h2h]    | ![alt text][roc_h2h]  |

[cm_h2h]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-h2h-form.png
[roc_h2h]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/roc-h2h-form.png

3. __Logistic Regression__

Best parameters:
```
LogisticRegression(C=0.002154434690031882, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='multinomial', n_jobs=1, penalty='l2',
          random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
          warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_lr]    | ![alt text][roc_lr]  |

[cm_lr]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-lr.png
[roc_lr]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/roc-lr.png

2. __Random Forest__

Best parameters:
```
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=15, n_jobs=1,
            oob_score=False, random_state=85, verbose=0, warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_rf]    | ![alt text][roc_rf] |

[cm_rf]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-rf.png
[roc_rf]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/roc-lr.png


3. __Gradient Boosting tree__

Best parameters:
```
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=100,
              presort='auto', random_state=0, subsample=1.0, verbose=False,
              warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_gbt]    | ![alt text][roc_gbt]  |

[cm_gbt]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-gbt.png
[roc_gbt]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/roc-gbt.png

4. __ADA boost tree__

```
AdaBoostClassifier(algorithm='SAMME',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=100, random_state=0)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_ada]   | ![alt text][roc_ada] |

[cm_ada]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-ada.png
[roc_ada]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/roc-ada.png

5. __Neural Net__

Best parameters:
```
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10, 5), learning_rate='constant',
       learning_rate_init=0.1, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=1e-10, validation_fraction=0.1, verbose=False,
       warm_start=False)
```
| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_nn]    | ![alt text][roc_nn]   |

[cm_nn]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-nn.png
[roc_nn]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-nn.png

6. __Light GBM__

Best parameters:
```
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=20,
        n_jobs=-1, num_leaves=31, objective=None, random_state=1,
        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)
```
| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_lgbm]    | ![alt text][roc_lgbm]   |

[cm_lgbm]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-lgbm.png
[roc_lgbm]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex1/cm-lgbm.png



__Experiment 2__

1. __Odd-based Decision Tree__:

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_odd]    | ![alt text][roc_odd]  |

[cm_odd]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-odd.png
[roc_odd]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/roc-odd.png

2. __h2h-Form-based Decision Tree__:

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_h2h]    | ![alt text][roc_h2h]  |

[cm_h2h]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-h2h-form.png
[roc_h2h]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/roc-h2h-form.png

2. __squad-strength-based Decision Tree__:

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_ss]    | ![alt text][roc_ss]  |

[cm_ss]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cam-ss.png
[roc_ss]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/roc-ss.png


1. __Logistic Regression__

Best parameters:
```
LogisticRegression(C=2.1544346900318823e-05, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='multinomial', n_jobs=1, penalty='l2',
          random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
          warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_lr]    | ![alt text][roc_lr]  |

[cm_lr]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-lr.png
[roc_lr]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/roc-lr.png

2. __Random Forest__

Best parameters:
```
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=15, n_jobs=1,
            oob_score=False, random_state=85, verbose=0, warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_rf]    | ![alt text][roc_rf] |

[cm_rf]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-rf.png
[roc_rf]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/roc-lr.png


3. __Gradient Boosting tree__

Best parameters:
```
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=1000,
              presort='auto', random_state=0, subsample=1.0, verbose=False,
              warm_start=False)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_gbt]    | ![alt text][roc_gbt]  |

[cm_gbt]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-gbt.png
[roc_gbt]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/roc-gbt.png

4. __ADA boost tree__

```
AdaBoostClassifier(algorithm='SAMME',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1, n_estimators=100, random_state=0)
```

| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_ada]   | ![alt text][roc_ada] |

[cm_ada]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-ada.png
[roc_ada]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/roc-ada.png

5. __Neural Net__

Best parameters:
```
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 15), learning_rate='constant',
       learning_rate_init=0.1, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=1e-10, validation_fraction=0.1, verbose=False,
       warm_start=False)
```
| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_nn]    | ![alt text][roc_nn]   |

[cm_nn]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-nn.png
[roc_nn]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-nn.png

6. __Light GBM__

Best parameters:
```
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        learning_rate=0.1, max_depth=-1, min_child_samples=20,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=15,
        n_jobs=-1, num_leaves=31, objective=None, random_state=1,
        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)
```
| Confusion matrix      |       ROC curve      |
|:---------------------:|:--------------------:|
| ![alt text][cm_lgbm]    | ![alt text][roc_lgbm]   |

[cm_lgbm]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-lgbm.png
[roc_lgbm]: https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/appendix/ex2/cm-lgbm.png
