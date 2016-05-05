import sentiment_mod as s

# print(s.sentiment("The show was good, I enjoyed it."))
# print(s.sentiment("I am filled with hatred."))

flag = "Y"

while flag == "Y" or flag == "y":
    ip = raw_input("Enter a string")
    print ip
    print "You are ", s.sentiment(ip)
    flag = raw_input("Enter y to continue")

print("Thanks for using!")
