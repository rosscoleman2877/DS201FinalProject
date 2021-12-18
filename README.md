# Examination of variables in order to Gather how Best to Predict Baseball Pitches

#### This repository was created by Ross Coleman, Saad Rehman, Lily Manetta, Maureen McCormack, and 

# Introduction
This project aims to use the steps of the entire data science process to gain actionable insight. For this project, we are analyzing the statistics of MLB pitcher Clayton Kershaw. We aim to walk users through the following steps: Business Understanding, Data Understanding, Data Preparation, Modeling + Evaluating (EDA + Machine Learning Model Training), and Deployment. This project aims to use the insights to help Clayton Kershaw be the best possible pitcher that he can be by exploiting the data.

# Business Understanding
In the United States, baseball is a very popular and competitive sport. In the MLB, an estimated 1,026 players compete. Outside of the MLB, making it to the MLB is a very competitive process. It is estimated that 10.5% of players advance to that level. In addition to the Major League (which is broadcasted all the time on ESPN and is what most people talk about), there are levels at which other players trying to make it to the MLB compete in. This league is referred to the Minor League. Ironically, a Lafayette alumni, JP Woodward, competes at this level. Post graduation from Lafayette, he signed with the Minor League Version of the Philadelphia Phillies. This next section of the business understanding will describe the differences between the two leagues as it is relevant for our analysis in that it validates the data we have and allows us to make sense of it from the standpoint of professional players.

In the Major League, there are two leagues- the American and the National. Each league has 15 teams and three divisions. Each team has what is known as a farm system which is where players and/or teams from the minor league are advanced up to the major league. Throughout the season, MLB teams will play not only the teams in their league but will also rotate and play teams outside of it. In terms of roster size, each team in the MLB carries 25 people, which is important as it shows the spots on these teams are extremely valuable and sought after. Ironically, just today, the MLB locked out its players. This is the first time this has happened in 26 years. A lockout occurs when the teams owners lock out its players. These issues typically stem from pay and items relating to the off-season as players are not paid. During lockouts, players are only paid through their signing bonuses and they do not receive their base salaries. This is relevant for our analysis as it shows how important it for Kershaw to use our actionable insight to be the best player he can be to receive a higher signing bonus.
  
   As baseball is measured on an inviduals performance, understanding those statistics is not only important to find ways to increase your performance but is also crucial as it shows players ways to not get cut. Already, baseball is a very difficult sport due to the nature of the sport; requiring players to catch/hit a small little ball, on an even smaller bat, is a hard task to accomplish. The sport requires incredible hand-eye coordination which to an extent, cannot be taught. Finding ways to increase ones natural talent is a difficult task so it is important to find ways to increase the non-natural talent of a player. 

  At the center of this project are the statistics of Clayton Kershaw. Kershaw has played 13 seasons with the Los Angeles Dodgers and he is a left-handed starting pitcher. Recently, Kershaw is a free agent which means that he is looking for another team to sign him. This role came with him finding out that the Dodgers were not offering him a Qualifying Offer. Talks of him being signed with the Boston Red Sox are in the works however no offers and signings are on the table. This signing for Kershaw is crucial in that it could be his last big-time contract as he is aging out and has earned more than enough income to survive. Spotrac estimated Kershaw earned a whopping $31 million dollars this past season. Additionally, Kershaw has been injured numerous times, which makes understanding his specific statistics so important. 

  For this project, we are analyzing Kershaw's pitching statistics in 2013. We chose to analyze this year as it had the most observations and was a year where Kershaw was not injured. For us, this makes analyzing his statistics easier as we can rule out other reasons for why he performed the way he did. Additionally in this year, Kershaw was a recipient of the Cy Young award, which was given to him for being the best pitcher in the National League. Understanding Kershaw's statistics are of great importance as it can help guide him to understand where he can be a better pitcher and help him to get a new contract with a highly-ranked team. 

  In addition to this project helping Kershaw to be the best pitcher he can be, our models can also help other pitchers understand what is most important about their pitch. We can help to answer the question of what aspects of a pitch most lead to a positive result for the pitcher. There are many different variables in our dataset that can help us with this, such as how fast the pitch is, how much spin is put on the pitch, and more. These results could change the way a pitcher trains, how much time he spends in the gym versus on the pitching mound, and what parts of his pitches he focuses on more.

  Baseball analytics is one of the fastest growing topics in the entire world. The use of analytics has revolutionized the baseball world to the point where now each baseball team has their own analytics department. This type of project is what each baseball team performs very often, all with the exact same goal: "What variables do we need to leverage in order to create the best baseball team we possibly can?
  
# Data Understanding
This data set contains pitch-by-pitch data for the MLB baseball player Clatyon Kershaw in the 2013 season. This data set contains 3,402 observations with 24 different variables. In this year, Clayton won the Cy Young award as the best pitcher in the National League. The variables within this dataset are measured using Major League Baseball's PITCHf/x system that uses camera systems in each ballpark to track characteristics of each pitch thrown.

The data was scraped from the MLB GameDay website (http://gd2.mlb.com/components/game/mlb/) using pitchRx

**BatterNumber:** Number of batters faced so far that game. The type of data for this variable is ratio. 

**Outcome:** One of 14 possible results for a pitch (e.g. Ball, Ball In Dirt, Called Strike, ..., Swinging Strike (Blocked). The data type for this variable is nominal.

**Class:** One of three classifications (B=ball, S=strike, or X=in play). The data type for this variable is nominal.

**Result:** From pitcher's perspective (Neg=ball or hit, Pos=strike or out).The data type for this variable is ordinal.

**Swing:** Did the batter swing at the pitch? (No or Yes). The data type for this variable is nominal. 

**Time:** Date and time of the pitch (format yyyy-mm-ddThh:mm:ssZ). The data type for this variable is interval. 

**StartSpeed:** Speed leaving the pitcher's hand (in mph). The data type of this variable is ratio. 

**EndSpeed:** Speed crossing home plate (in mph). The data type for this variable is ratio. 

**HDev:** Horizontal movement (inches). The data type for this variable is ratio. 

**VDev:** Vertical movement (inches). The data type for this variable is ratio. 

**HPos:** Horizontal position at home plate (inches from center, positive is catcher's right). The data type for this variable is ratio. 

**VPos:** Vertical position at home plate (inches above the ground).The data type for this variable is ratio. 

**PitchType:** Code for pitch type (CH=changeup, CU=curve, FF=fastball, or SL=slider). The data type for this variable is nominal. 

**Zone:** 1-9 in theoretical strike zone (upper left to lower right), 11-14 are out of strike zone. The data type for this variable is ordinal. 

**Nasty:** A measure on a 0-100 scale of difficulty of the pitch to hit (100 is most difficult).The data type for this variable is ratio. 

**Count:**Ball strike count (0-0, 0-1, 0-2, 1-1, 1-2, 2-1, 2-2, 3-1, or 3-2).The data type for this variable is ordinal. 

**BallCount:** Number of balls before the pitch (0, 1, 2, or 3). The data type for this variable is ratio. 

**StrikeCount:** Number of strikes before the pitch (0, 1, or 2). The data type for this variable is ratio. 

**Inning:** Inning of the game. The data type for this variable is ordinal. 

**InningSide:** Portion of the inning (bottom= pitcher at home or top=pitcher away). The data type for this variable is nominal. 

**Outs:** Number of outs when the pitch is thrown. The data type for this variable is ratio. 

**BatterHand:** Batter's stance (L=left or R=right). The data type for this variable is nominal. 

**ABEvent:** Result of the at bat (several possibilities). The data type for this variable is nominal. 

**Batter:** Name of the batter faced. The data type for this variable is nominal. 


**Source:** Data scraped from the MLB GameDay website (http://gd2.mlb.com/components/game/mlb/) using pitchRx

# Modeling and Evaluation
### Exploratory Data Analysis
<img scr ="plot 1.png" width="600" height="400">
The histogram shows the inning of the game and the frequency of the pitches Kershaw threw. From this we can see that Kershaw threw the majority of his pitches between the third and fourth inning and his least pitches in the eighth and ninth. Additionally, from the fifth inning, we can see that his pitching decreases.
                                          
