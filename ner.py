from TwitterAPI import TwitterAPI

consumer_key = 'WzQ61Y2dpCDP5PL2Inq5J4rMr'
consumer_secret = 'icPhj44wtybuKYKHU5tTtDyP4MRtvBqhAyDDPglzvOX0pUwNNz'
access_token_key = 	'985292614000771072-oTUNGgVHJW81FuE5C87jZkPKAJjAGhm'
access_token_secret = 'LyWL8yKorFVk3lwjTrFb9FnoeatRlISUxiYbOG9xVwvq6'

api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret)

r = api.request('search/tweets', {'q':'pizza'})
for item in r:
        print(item)