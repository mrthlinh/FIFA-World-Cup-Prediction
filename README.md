# Project Description
__Objective__: Prediction of the winner of FIFA world cup 2018. Prediction results are  Win / Lose / Draw

__Data__: Data are assembled from multiple sources, most of them are from Kaggle, others come from FIFA website / EA games and I need to build a data crawler.

__Feature Selection__: To determine who will more likely to win a match, based on my knowledge, I come up with 4 main groups of features as follows:
1. head-to-head match history
2. recent performance of each team (10 recent matches), aka "form"
3. bet-ratio before matches
4. squad value or how many players are in top 200 (from video game)
Feature list reflects those factors.

__Supervisor__: Pratibha Rathore

__Lifecycle__

![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/life_cycle.png)

# Data
### Data Source
The dataset are from all international matches from 2000 - 2017, results, bet odds, ranking, title won.
1. [FIFA World Cup 2018](https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data)
2. [International match 1872 - 2017](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data)
3. [Ranking through Time](http://www.fifa.com/fifa-world-ranking/associations/association=usa/men/index.html)
4. [Bet Odd](https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data)
5. [Player Rating Through time](https://www.futhead.com/10/players/?page=2)
6. [Squad for each tournament](https://github.com/openfootball/world-cup)


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
    For this question, we use the average odd getting from [Bet Odd][2] before matches.

    ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/avg_odd_win.png)


    ```python
    Ttest_indResult(statistic=-43.22461132125629, pvalue=0.0)
    ```

    We can say that we can reply on bet odd to predict the match results.

# Model

Model we used and evaluation criteria
### Machine learning models
1. Logistic Regression
  - Best parameters:
  ```python
  LogisticRegression(C=0.00046415888336127773, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='multinomial', n_jobs=1, penalty='l2',
          random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
          warm_start=False)
  ```
  - Confusion matrix, table and plot

  ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/cm_lr.png)  
  - ROC curve and Area Under Curve
  ![](https://github.com/mrthlinh/FIFA-World-Cup-Prediction/blob/master/pic/roc_lr.png)  

2. SVM with Gaussian Kernel
3. Naive Bayes
4. Decision Tree
5. Random Forest
6. Deep Learning

### Train-Test
- Train set:
- Test set : EURO 2016

### Evaluation Criteria
- Precision
- Recall
- F1
- accuracy
- ROC curve

# Results

# Reference
1. [A machine learning framework for sport result prediction](https://www.sciencedirect.com/science/article/pii/S2210832717301485)
2. [t-test definition](https://en.wikipedia.org/wiki/Student%27s_t-test)
3. [Confusion Matrix Multi-Label example](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
4. [Precision-Recall Multi-Label example](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#in-multi-label-settings)
5. [ROC curve example](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)
6. [Model evaluation](http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
7. [Tuning the hyper-parameters of an estimator](http://scikit-learn.org/stable/modules/grid_search.html)
8. [Validation curves](http://scikit-learn.org/stable/modules/learning_curve.html)
# Task List
__Ongoing__
- [ ] Add feature group 1
    - [x] Add h_win_diff, h_draw
    - [ ] Add rank_diff, title_diff
- [ ] Migrate code of pre-processing plot function to python file with Spyder
- [ ] Build a data without player rating and squad value
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
