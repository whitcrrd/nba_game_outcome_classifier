# NBA Win/Loss Classifier

### Underlying Question: 
Can we accurately classify NBA games as won/lost using statistics through the first three quarters? 

### Data Background: 
Each team in the NBA plays 82 games a year, and between all NBA teams, there are a total of 1,230 games per year (ignoring lockout).  So, to get enough data, we'll be pulling games from multiple seasons (7).  The focus of our classifier will be on predicting whether the **home team** wins using only data from the first three quarters.

### The Dataset: 
All NBA regular season games from the last seven seasons (2012 - 2019), sourced from [stats.nba](https://stats.nba.com) JSON endpoint.  Each game instance returned from the JSON endpoint has extraneous statistics and attributes, so when loading the JSON and creating our dataset, we hand pick only a couple of traditional metrics.

## Data Obtainment & Preprocessing

Since the NBA returns traditional and advanced statistics from two different endpoints, we only use the traditional statistics endpoint, and calculate the advanced metrics from this.  Additionally, the nba.stats.com API returns 1st Half & individual quarters for an individual season from different endpoints, and returns two rows for each NBA game, one for the home team and one for the away team.  

Ultimately, I would like to have single dataframe that:

- Has the "End of Third Quarter" Statistics (from two different endpoints)
- Has Team & Opponent Statistics (from two rows in each endpoint)

To get here, I'll have to:

- **Import Half Time + 3rd Quarter** stats for a single season

- **Drop extra columns** that nba.stats.com returns (Season Ranks, Team Names, Game Times, etc.)

- **Rename columns** in each dataframe to reflect which stat we're working with ("HALF_", "3Q_")

- **Combine Half Time + 3rd Quarter** dataframes into a single dataframe, on their matching GAME_ID and TEAM_ID.

- **Add "HALF_" + "3Q_"** stat columns to get "End of 3rd Quarter" stats

- **Drop "HALF_" + "3Q_"** stat columns

- **Recalculate statistics** (FG%, FG3%, FT%)

- **Set "HOME" column** (binary) for rows (Matchup field contains 'vs.' if home)

- **Sort dataframe** by GAME_ID, since there are two rows per game (home and away)

- **Iterate dataframe** and add away team's stats to home team rows as "OPP_[STAT]"

- **Drop away team rows** (since all the statistics are now in the home team rows)

- **Calculate four factor** (advanced NBA metrics) statistics for home and opponent stats

- **Drop columns** used in feature engineering four factor metrics

## EDA - Distribution of Shooting Statistics in W/L

As assumed, shooting statistics correlate to game outcomes, with winning games having slightly higher effective field goal averages and lower opponent effective field goal averages.  Additionally, when computing the NET effective field goal distribution, this difference becomes incredibly more evident, as the two distributions move further apart.  

#### Opponent Effective Field Goal %

![OPP EFG distribution](https://i.imgur.com/6jYLL9p.png)

#### Effective Field Goal % (Home Team)

![EFG distribution](https://i.imgur.com/oltJgXm.png)

#### NET Effective Field Goal %

![NET EFG distribution](https://i.imgur.com/On4UCmb.png)


Additionally, the notebook includes interactive pie charts to visualize game outcomes by category and min/max values.  For example, effective field goal shooting:

![pie](https://i.imgur.com/HddArQY.gif)


#### Feature Correlation - Heat Map

![heat map](https://i.imgur.com/08GgicM.png)


## Models

#### Cross Validation & Accuracy Distribution

I applied three different model architectures on the dataset - (1) Random Forest, (2) Decision Tree, and (3) SVM.  Here's a quick look at the accuracy distributions of each: 

![accuracy distribution](https://i.imgur.com/mMuP6e2.png)


Of the three models, Decision Tree was noticeably the worst performing model. Random Forest & SVM had similar mean accuracy scores, with SVM having a slightly higher accuracy and more consistent results (lower standard deviation). Next, I'll want to use grid search to find the optimal hyperparameters for Random Forest & SVM.


#### SVM vs. Random Forest

![models](https://i.imgur.com/xUkl8Z9.png)

Both models, after hyperparameter tuning, ended up with ~82% accuracy scores, and almost identical false positive, true positive, false negative, true negative rates.  Given that the results are nearly identical, I would be fine working with either model going forward.

#### Confusion Matrix (Final Model)

**Precision** = True Positive / (True Positive + False Positive)

Or, Total True Positive / Total Predicted Positive.  Precision is preferred when the cost of a False Positive is high, such as spam.  A wrongly classified work email that hits your spam folder is worse than an errant spam email making it into your inbox.

**Recall** = True Positive / (True Positive + False Negative)

Or, True Positive / Total Actual Positive.  Recall measures the accuracy of how many of the actual positives the model labels as positive.  Recall is preferred when the cost of a False Negative is high, such as when diagnosing medical conditions.

**F1 Score** = 2 * (Precision * Recall)/(Precision + Recall)

![confusion matrix](https://i.imgur.com/h2cK6Hc.png)

The final model had an 87% True Positive rate, 75% True Negative rate, 25% False Positive Rate, and 13% False Negative rate.  Not particularly outstanding results, but we have to remember that, in close games, a single possession can change the outcome of the game.  Additionally, for future improvements, I would add NET effective field goal % and would restructure the original question.



#### Feature Importance (Random Forest)

![feature importance](https://i.imgur.com/QPM1JX3.png)


In the Random Forest classifier, the most important feature was PlUS_MINUS, followed by shooting statistics.  This is intuitive, since, if you're losing through three quarters (75% of the game), you have to have a 4th quarter that diverges from the losing trend set by the first three quarters.  And, in doing so, it doesn't necessarily equate into a win, just a closer game.  Additionally, the 4 least important features (DREB %, OPP BLK, FT RATE, & OPP FT RATE) were removed in the final model without any reduction in accuracy.

## Final Thoughts

In hindsight, including NET effective field goal instead of effective field goal & opponent effective field goal would produce better results.

If I had more time, I would have liked to re-structure the initial question and try to predict ranges of game outcomes in terms of PLUS_MINUS.  That is, rather than predicting a win or a loss, I would focus on creating a model that could predict ranges of PLUS_MINUS (ex, <5 points is a "very close" game, 5 < x < 10 points is an "average" game, and > 10 points would be a clear win).  

Additionally, I would be interested in creating a model that either makes a prediction given a confidence level, or abstains from predicting.  That is, I would rather have the model reveal "Games in which the winner is X 95% of the time", rather than predicting games that are hard to classify.

![final thoughts slide](https://i.imgur.com/eLNnmpZ.png)