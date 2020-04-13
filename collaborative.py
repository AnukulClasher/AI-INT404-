import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.distance import correlation

dataimp=pd.read_csv('dataimp4_1.csv')
placeInformation=pd.read_csv('dataimp4.csv')

dataimp=pd.merge(dataimp,placeInformation,left_on='itemId',right_on="itemId")
userId=dataimp.userId
userId2a=dataimp[['userId']]

dataimp.loc[0:10,['userId']]
dataimp=pd.DataimpFrame.sort_values(dataimp,['userId','itemId'],ascending=[0,1])


def favoritePlaces(activeUser,N):
    topPlaces=pd.DataimpFrame.sort_values(
        dataimp[dataimp.userId==activeUser],['rating'],ascending=[0])[:N]
    return list(topPlaces.title)

userItemRatingMatrix=pd.pivot_table(dataimp, values='rating',
                                    index=['userId'], columns=['itemId'])


def similarity_function(user1,user2):
    try:
        user1=np.array(user1)-np.nanmean(user1)
        user2=np.array(user2)-np.nanmean(user2)
        commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
        if len(commonItemIds)==0:
           return 0
        else:
           user1=np.array([user1[i] for i in commonItemIds])
           user2=np.array([user2[i] for i in commonItemIds])
           return correlation(user1,user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")



def nearestNeighbourRatingstovisit(activeUser,K):
    try:
        similarity_fun_Matrix=pd.DataimpFrame(index=userItemRatingMatrix.index,columns=['Similarity_function'])
        for i in userItemRatingMatrix.index:
            similarity_fun_Matrix.loc[i]=similarity_function(userItemRatingMatrix.loc[activeUser],userItemRatingMatrix.loc[i])
        similarity_fun_Matrix=pd.DataimpFrame.sort_values(similarity_fun_Matrix,['Similarity_function'],ascending=[0])
        nearestNeighbours=similarity_fun_Matrix[:K]
        neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
        predictItemRating=pd.DataimpFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
        for i in userItemRatingMatrix.columns:
            predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if userItemRatingMatrix.loc[j,i]>0:
                   predictedRating += (userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity_function']
                predictItemRating.loc[i,'Rating']=predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")            
    return predictItemRating


def topNRecommendations_func(activeUser,N):
    try:
        predictItemRating=nearestNeighbourRatingstovisit(activeUser,10)
        placeAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
        predictItemRating=predictItemRating.drop(placeAlreadyWatched)
        topRecommendations=pd.DataimpFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
        topRecommendationTitles=(placeInformation.loc[placeInformation.itemId.isin(topRecommendations.index)])
    except ZeroDivisionError:
        print("You can't divide by zero!")
    return list(topRecommendationTitles.title)


activeUser=int(input("Enter userid: "))
#print("The user's favorite places are: ")
#print(favoritePlaces(activeUser,5))
print("The recommended places for you are: ")
print(topNRecommendations_func(activeUser,4))
