# FIFA World Cup Prediction
Objective: Prediction of the winner of FIFA world cup 2018. I focus on predict result (Win/Lose/Draw) each match, my predictor also get updated according to the actual results for each match.

Data: Data are assembled from multiple sources, most of them are frome Kaggle, others come from FIFA website / EA games and I need to build a data crawler.
 
Feature Selection: To determine who will more likely to win a match, I come up with 4 main factors as follows:
- head-to-head match history
- recent performance of each team (10 recent matches)
- bet-ratio before the match
- squad value or how many players are in top 100 (from video game)

Feature list reflects those factors. 
### Feature List

| Feature Name  | Description                   | Source | Status |
|:-------------:|:-----------------------------:|:------:|:------:|
| Team 1        | Nation Code (e.g US, NZ)      | 1 ||
| Team 2        | Nation Code  (e.g US, NZ)     | 1 ||
| Date          | Date of match                 |2||
| Tournament    | Friendly,EURO, AFC, FIFA WC   |2||
| bet_ratio     | ratio of bet                  | 4 ||
| Rank_Diff     | #Rank T1 - T2                 |3|Crawler|
| Title_Diff    | #Title won T1 - T2            |FIFA Web|Crawler|
| Win_Diff      | #Win T1 - T2                  |2||
| Draw          | #Draw                         |2||
| L_Num_Goal    | #Goal in 10 recent matches    |2||
| L_win         | #Win in 10 recent matches     |2||
| L_draw        | #draw in 10 recent matches    |2||

### Dataset:
The dataset are from all international matches from 2000 - 2017, results, bet odds, ranking, title won.
1. [FIFA World Cup 2018](https://www.kaggle.com/ahmedelnaggar/fifa-worldcup-2018-dataset/data)
2. [International match 1872 - 2017](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data)
3. [Ranking through Time](http://www.fifa.com/fifa-world-ranking/associations/association=usa/men/index.html)
4. [Bet Odd](https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset/data)
5. [Player Rating Through time](https://www.futhead.com/10/players/?page=2)
6. [Squad for each tournament](https://github.com/openfootball/world-cup)

### Train-Test
- Train set: International matches from 2000 - 2017 (exclude EURO 2016 and World Cup 2014)
- Test set : EURO 2016 and World Cup 2014

### Reference
1. [A machine learning framework for sport result prediction](https://www.sciencedirect.com/science/article/pii/S2210832717301485)

### Task List
- [ ] Add features reflects head-to-head stats
- [ ] Add features reflects recent form (in 10 recent matches) of 2 teams
- [ ] Add features of bets / difference, ratio of bet before match
- [ ] Add features: Squad Values, Number of Stars, Sum of player ratings
- [ ] Build a data without player rating and squad value
- [ ] Build a web crawler for Ranking over time
- [ ] A table of title won for each team
- [ ] Integrate player rating and squad value to data


