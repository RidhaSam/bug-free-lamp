import random as rd

suitList = ["♥","♠","♦","♣"]
rankList = ["2","3","4","5","6","7","8","9","10","K","Q","J","A"]

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.rank + self.suit

    def getValue(self):
        if self.rank in  rankList[:9]:
            return int(self.rank)
        elif (self.rank == "K"):
            return 13
        elif (self.rank == "Q"):
            return 12
        elif (self.rank == "J"):
            return 11
        else:
            return 14

    def __lt__(self,other):
        return (self.getValue() < other.getValue())
    
    def __gt__(self,other):
        return (self.getValue() > other.getValue())
    
    def __eq__(self,other):
        return (self.getValue() == other.getValue())

class Deck:
    def __init__(self):
        self.cardDeck = []
        for suit in suitList:
            for rank in rankList:
                self.cardDeck.append(Card(suit,rank))

    def __str__(self):
        retString = ""
        for card in self.cardDeck:
            retString = retString + " " + str(card)
        return retString

    def shuffle(self):
        rd.shuffle(self.cardDeck)

    def removeCard(self,card):
        if card in self.cardDeck:
            self.cardDeck.remove(card)
            return True
        else:
            return False

    def takeCard(self):
        return self.cardDeck.pop(0)

    def dealCards(self, hand, nCards):
            for i in range(nCards):
                if (len(self.cardDeck) == 0):
                    break
                else:
                    card = self.takeCard()
                    hand.addCard(card)

class Hand(Deck):
    def __init__(self, name=""):
        self.cardDeck = []
        self.name = name

    def addCard(self,card):
        self.cardDeck.append(card)

    def cardsLeft(self):
        return len(self.cardDeck)

def warCompareCards(warCards, playerHand, opponentHand):
    print("Your card: " + str(warCards[0]) + " Opponent's Card: " + str(warCards[1]) + "\n")
    if (warCards[0] > warCards[1]):
        for card in warCards:
            playerHand.addCard(card)
        print("You win the round!\n")
    elif (warCards[0] < warCards[1]):
        for card in warCards:
            opponentHand.addCard(card)
        print("You lost the round!\n")
    else:
        print("WAR!\n")
        if (playerHand.cardsLeft() >= 4 and opponentHand.cardsLeft() >= 4):
            for i in range(4):
                warCards.insert(0,opponentHand.takeCard())
                warCards.insert(0,playerHand.takeCard())
            warCompareCards(warCards, playerHand, opponentHand)
        else:
            while ((playerHand.cardsLeft() > 0) and (opponentHand.cardsLeft() > 0)):
                warCards.append(playerHand.takeCard())
                warCards.append(opponentHand.takeCard())

def warGame():
    print("Welcome to War \n")
    playerName = input("Enter your name: ")
    opponentName = input("Enter your opponent's name: ")
    playerHand = Hand(playerName)
    opponentHand = Hand(opponentName)
    
    gameDeck = Deck()
    gameDeck.shuffle()
    gameDeck.dealCards(playerHand, 26)
    gameDeck.dealCards(opponentHand, 26)
    print("The game will begin now")
    while ((playerHand.cardsLeft() > 0) and (opponentHand.cardsLeft() > 0)):
        print("You have ", playerHand.cardsLeft(), " cards")
        resp = input("Enter 'Q' to quit, or any other key to place your card.\n")
        if (resp.upper() == 'Q'):
            break
        else:
            playerCard = playerHand.takeCard()
            opponentCard = opponentHand.takeCard()
            warCards = [playerCard, opponentCard]
            warCompareCards(warCards, playerHand, opponentHand)

    if (playerHand.cardsLeft() == 0):
        print("You lost")
    elif (opponentHand.cardsLeft() == 0):
        print("You win!")
    else:
        print("You have quit")
                    
warGame()
    


