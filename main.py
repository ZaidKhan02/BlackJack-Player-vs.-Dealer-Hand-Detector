from ultralytics import YOLO
import cv2
import cvzone

# Initialize the webcam for capturing real-time video feed
cap = cv2.VideoCapture(0)  # Capture video from the default webcam
cap.set(3, 1280)  # Set the width of the video frame to 1280 pixels for better resolution
cap.set(4, 720)  # Set the height of the video frame to 720 pixels for better resolution

# Load the YOLO model trained to detect playing cards
model = YOLO("playingCards.pt")  # "playingCards.pt" is the pre-trained CNN model file

# Here we define the list of card class names. Each name corresponds to a class in the YOLO model
# The names represent standard playing cards with their values (e.g., '10C' for Ten of Clubs). The index of
# the cards on this list represent the integer that the model uses to classify cards in the training labels
classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

# Function to calculate the total value of a hand in Blackjack
# Takes a list of card identifiers (e.g. '5H', 'AD') and calculates the total hand value
# Handles special Blackjack rules for Aces (worth 11 or 1 depending on the total)
def calculate_blackjack_value(hand):
    value = 0  # Initialize the total hand value
    aces = 0  # Count the number of Aces in the hand

    # Loop through each card in the hand
    for card in hand:
        rank = card[:-1]  # Extract the value of the card (e.g., '10' from '10H')
        if rank in ['K', 'Q', 'J']:  # Face cards (King, Queen, Jack) are worth 10 points
            value += 10
        elif rank == 'A':  # Aces are initially worth 11 points
            aces += 1
            value += 11
        else:  # Number cards (e.g., '2', '3') are worth their face value
            value += int(rank)

    # Adjust the value of Aces to 1 if the total exceeds 21
    while value > 21 and aces > 0:
        value -= 10  # Reduce the value of an Ace from 11 to 1
        aces -= 1

    return value  # Return the calculated hand value

# Variables to keep track of the game state
game_over = False  # Indicates whether the game has ended
winner_message = ""  # Stores the message declaring the winner

# Main loop to process the video feed frame by frame
while True:
    success, img = cap.read()  # Read a frame from the webcam
    results = model(img, stream=True)  # Use the YOLO model to detect cards in the frame

    # Lists to store detected cards for the dealer and player
    dealer_hand = []
    player_hand = []

    # Initialize default hand values
    dealer_value = 0
    player_value = 0

    # Get the dimensions of the video frame for zone definitions
    height, width, _ = img.shape
    dealer_zone = (0, 0, width, height // 2)  # Top half of the frame is the dealer's card zone
    player_zone = (0, height // 2, width, height)  # Bottom half of the frame is the player's card zone

    # Draw a dividing line to visually separate the two zones using cv2
    separator_y = height // 2
    cv2.line(img, (0, separator_y), (width, separator_y), (0, 255, 255), 3)

    # Loop over each result from the model inference
    for r in results:
        # Retrieve the detected bounding boxes for objects (cards) in the current frame
        # 'boxes' is an attribute of the YOLO detection results object 'r', which contains information
        # about all the bounding boxes detected in the frame, including their coordinates and class IDs
        boxes = r.boxes

        # Loop through each detected bounding box to process its information
        for box in boxes:
            #
            # 'box.xyxy[0]' provides the top-left (x1, y1) and bottom-right (x2, y2) coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            # Convert the coordinates from floating-point values to integers for easier use in drawing and calculations
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Retrieve the class index (representing the type of card detected) from the YOLO result
            # 'box.cls[0]' gives the class ID as a float, which is converted to an integer for indexing
            cls = int(box.cls[0])
            # Map the class index to its corresponding card name using the 'classNames' list
            # For example, if 'cls' is 10, the corresponding card name might be '3C' (Three of Clubs)
            card = classNames[cls]

            # Determine the zone (dealer's or player's) in which the detected card is located.=
            # This decision is based on the vertical position (y-coordinate) of the top-left corner of the bounding box
            if y1 < dealer_zone[3]:  # Check if the card's top is within the dealer's zone (top half of the frame)
                dealer_hand.append(card)  # Add the card to the dealer's hand list.=
            elif y1 >= player_zone[1]:  # Check if the card's top is within the player's zone (bottom half of the frame)
                player_hand.append(card)  # Add the card to the player's hand list

            # Draw a green bounding box around the detected card on the video frame
            # This provides a visual representation of where the card is located in the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add a text label (the name of the detected card) near the bounding box for clarity.
            # The label is placed slightly above the top-left corner of the bounding box (y1 - 10)
            label = f"{card}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Remove any duplicate cards from the dealer's and player's hands to ensure each card is counted only once
    # The 'set' function removes duplicates, and the result is converted back to a list
    dealer_hand = list(set(dealer_hand))
    player_hand = list(set(player_hand))

    # If the game is ongoing, calculate and display hand values
    if not game_over:
        if dealer_hand:  # Calculate and display dealer's hand value if cards are present
            dealer_value = calculate_blackjack_value(dealer_hand)
            dealer_result = f"Dealer: {'BUST' if dealer_value > 21 else dealer_value}"
            cvzone.putTextRect(img, dealer_result, (50, 50), scale=2, thickness=2, colorR=(0, 0, 0),
                               colorB=(255, 255, 255))

        if player_hand:  # Calculate and display player's hand value if cards are present
            player_value = calculate_blackjack_value(player_hand)
            player_result = f"Player: {'BUST' if player_value > 21 else player_value}"
            cvzone.putTextRect(img, player_result, (50, separator_y + 50), scale=2, thickness=2, colorR=(0, 0, 0),
                               colorB=(255, 255, 255))

    # Display the winner message if the game is over
    if game_over:
        cvzone.putTextRect(
            img,
            winner_message,  # Center the winner message in the frame
            (width // 2 - 200, height // 2 - 50),
            scale=3,
            thickness=3,
            colorR=(0, 0, 0),
            colorB=(255, 255, 255)
        )

    # Display the annotated video feed in a window
    cv2.imshow("Image", img)

    # Handle keyboard inputs to allow dealer to control the game
    key = cv2.waitKey(1) & 0xFF

    if key == ord('S'):  # End the game when 'shift+s' is pressed
        game_over = True
        # Determine the winner based on hand values.
        if dealer_value > 21 and player_value > 21:
            winner_message = "Both players busted! No winner."
        elif dealer_value > 21:
            winner_message = "Dealer busted! Player wins!"
        elif player_value > 21:
            winner_message = "Player busted! Dealer wins!"
        elif dealer_value > player_value:
            winner_message = "Dealer wins!"
        elif player_value > dealer_value:
            winner_message = "Player wins!"
        else:
            winner_message = "It's a draw!"

    if key == ord('D'):  # Resets the game when 'shift+d' is pressed
        game_over = False
        winner_message = ""

    if key == ord('q'):  # This will exit the program when 'q' is pressed
        break
