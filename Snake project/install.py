import pygame
import random
import cv2 # Computer Vision
import mediapipe as mp # Hand Tracking
import threading # To run vision and game simultaneously

# --- GLOBAL CONTROL VARIABLE ---
# Shared variable to hold the direction command from the vision thread
VISION_DIRECTION = (1, 0)  # Default: Right

# --- 1. Game Constants ---
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE
FPS = 10 

# --- 2. Colors and Pygame Setup (Same as before) ---
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Gesture-Controlled Snake")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# --- 3. Snake and Food Setup (Same as before) ---
def reset_game():
    global VISION_DIRECTION
    snake = [(GRID_WIDTH // 4, GRID_HEIGHT // 2)]
    # Use the global variable for direction
    direction = VISION_DIRECTION 
    food = generate_food(snake)
    score = 0
    return snake, direction, food, score

def generate_food(snake):
    while True:
        food_pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if food_pos not in snake:
            return food_pos

def draw_element(surface, color, pos):
    rect = pygame.Rect(pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    pygame.draw.rect(surface, color, rect)

def draw_score(surface, score):
    text = font.render(f"Score: {score}", True, WHITE)
    surface.blit(text, (5, 5))

# --- 4. Computer Vision (The Gesture Control) Thread ---

def vision_thread_func():
    global VISION_DIRECTION
    
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Webcam setup
    cap = cv2.VideoCapture(0)
    
    # State tracking variables for gesture
    # We will track the X, Y position of the Index Finger Tip (Landmark 8)
    initial_pos = None 
    tracking_threshold = 30 # Minimum pixel movement for a gesture
    
    # The loop for reading the webcam
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a natural, mirror-like view
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = hands.process(image_rgb)
        
        # Get image dimensions
        h, w, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the tip of the Index Finger (Landmark 8)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                current_x = int(index_tip.x * w)
                current_y = int(index_tip.y * h)
                
                # --- Gesture Recognition Logic ---
                if initial_pos is None:
                    # Start tracking when a hand is detected
                    initial_pos = (current_x, current_y)
                
                # Calculate movement vector
                dx = current_x - initial_pos[0]
                dy = current_y - initial_pos[1]
                
                # Check if a significant movement has occurred (a 'swipe')
                if abs(dx) > tracking_threshold or abs(dy) > tracking_threshold:
                    
                    # Determine the main direction of movement
                    if abs(dx) > abs(dy): # Horizontal movement is dominant
                        if dx > 0:
                            new_direction = (1, 0) # Right
                        else:
                            new_direction = (-1, 0) # Left
                    else: # Vertical movement is dominant
                        if dy > 0:
                            new_direction = (0, 1) # Down
                        else:
                            new_direction = (0, -1) # Up

                    # Only update if the direction is not opposite to the current one
                    current_dx, current_dy = VISION_DIRECTION
                    if (new_direction[0] != -current_dx) or (new_direction[1] != -current_dy):
                        VISION_DIRECTION = new_direction
                    
                    # Reset initial position after a successful gesture to wait for the next one
                    initial_pos = (current_x, current_y) 
                    
                # Optional: Draw a circle at the tracked point
                cv2.circle(image, (current_x, current_y), 10, (0, 255, 0), -1)

        # Display the webcam feed in a separate window
        cv2.imshow('Gesture Control Feed', image)
        
        # Exit vision thread if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


# --- 5. Main Game Loop (Modified) ---
def game_loop():
    global VISION_DIRECTION
    
    # 1. Start the Vision Thread
    vision_thread = threading.Thread(target=vision_thread_func)
    vision_thread.daemon = True # Allows program to exit if main thread stops
    vision_thread.start()
    
    snake, direction, food, score = reset_game()
    running = True
    game_over = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Allow SPACE to restart
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and game_over:
                snake, direction, food, score = reset_game()
                game_over = False
                
        # --- CONTROL INTEGRATION: Use the Direction from the Vision Thread ---
        if not game_over:
            # Snake's direction is NOW set by the VISION_DIRECTION global variable
            direction = VISION_DIRECTION
            
            # Rest of the game logic (movement, collision, food) is the same
            head_x, head_y = snake[0]
            new_head = (head_x + direction[0], head_y + direction[1])

            # Check for wall collision
            if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
                new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
                game_over = True
            
            # Check for self-collision
            if new_head in snake:
                game_over = True

            snake.insert(0, new_head)

            # Check for food
            if new_head == food:
                score += 1
                food = generate_food(snake)
            else:
                snake.pop() 

        # --- Drawing (Same as before) ---
        screen.fill(BLACK)
        for pos in snake:
            draw_element(screen, GREEN, pos)
        draw_element(screen, RED, food)
        draw_score(screen, score)

        if game_over:
            game_over_text = font.render("Game Over! Press SPACE to Restart or 'q' in video feed to Quit.", True, WHITE)
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2))

        pygame.display.flip()
        clock.tick(FPS) 

    pygame.quit()
    cv2.destroyAllWindows() # Make sure to close all OpenCV windows too!

if __name__ == "__main__":
    game_loop()