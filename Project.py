import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import os
import pickle
from skimage.transform import resize
from skimage.feature import hog
import re
from tkinter import messagebox
import pandas as pd
import sys 
from tkinter import * 
from tkinter import ttk
import tkinter as tk
from tkinter.constants import LEFT, TOP

#Reading and displaying original image
f = plt.figure(figsize=(8,8))

window_name = 'image'
img = cv2.imread('sample1.jpg')
height, width, depth = img.shape
f.add_subplot(3,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

#Applying Gaussian Blurring to remove noise from the image
blur = cv2.GaussianBlur(img,(3,3),0)
f.add_subplot(3,3,2)
plt.imshow(blur)
plt.title("Blurring")
plt.axis('off')

#Applying Thresholding to account for different illuminations in different parts of the image.
img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 199, 5)
f.add_subplot(3,3,3)
plt.imshow(thresh2, cmap="gray")
plt.title("Thresholding")
plt.axis('off')

#Inverting to make the digits and lines white while making the background black.
inverted_image = cv2.bitwise_not(thresh2)
f.add_subplot(3,3,4)
plt.imshow(inverted_image, cmap="gray")
plt.title("Inverted")
plt.axis('off')

#Dilation with a 3x3 shaped Kernel to fill out any cracks in the board lines and thicken the board lines.
kernel = np.ones((3,3), np.uint8)
img_dilation = cv2.dilate(inverted_image, kernel, iterations=1)
f.add_subplot(3,3,5)
plt.imshow(img_dilation, cmap="gray")
plt.title("Dilating to Fill up Cracks")
plt.axis('off')

#Flood Filling Since the board will probably be the largest blob a.k.a connected component with the largest area, floodfilling from different seed points and finding all connected components followed by finding the largest floodfilled area region will give the board.
outerbox = img_dilation
maxi = -1
maxpt = None
value = 10
height, width = np.shape(outerbox)
for y in range(height):
    row = outerbox[y]
    for x in range(width):
        if row[x] >= 128:
            area = cv2.floodFill(outerbox, None, (x, y), 64)[0]
            if value > 0:
                value -= 1
            if area > maxi:
                maxpt = (x, y)
                maxi = area

# Floodfill the biggest blob with white (Our sudoku board's outer grid)
cv2.floodFill(outerbox, None, maxpt, (255, 255, 255))

# Floodfill the other blobs with black
for y in range(height):
    row = img_dilation[y]
    for x in range(width):
        if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
            cv2.floodFill(outerbox, None, (x, y), 0)

f.add_subplot(3,3,6)
plt.imshow(outerbox, cmap="gray")
plt.title("Flood Filling")
plt.axis('off')

# Eroding it a bit to restore the image
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
outerbox = cv2.erode(outerbox, kernel)
f.add_subplot(3,3,7)
plt.imshow(outerbox, cmap="gray")
plt.title("Eroding the grid")
plt.axis('off')

#HoughLine Transformtion
edges = cv2.Canny(outerbox,50,150,apertureSize=3)
im = outerbox.copy()

rho, theta, thresh = 2, np.pi/180, 400
lines = cv2.HoughLines(edges, rho, theta, thresh)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        #r * cost(theta) - 1000 * sin(theta) 
        x1 = int(x0 + 3000*(-b))
        #r * cost(theta) + 1000 * sin(theta)
        y1 = int(y0 + 3000*(a))
        #r * cost(theta) + 1000 * sin(theta)
        x2 = int(x0 - 3000*(-b))
        #r * cost(theta) - 1000 * sin(theta)
        y2 = int(y0 - 3000*(a))
        #(x1,y1) starting points, (x2,y2) ending points
        cv2.line(im,(x1,y1),(x2,y2),(0,255,0),10)

f.add_subplot(3,3,8)
plt.imshow(im, cmap="gray")
plt.title("Houghline Transformation")
plt.axis('off')


def mergeLines(lines, img):
            height, width = np.shape(img)
            for current in lines:
                if current[0][0] is None and current[0][1] is None:
                    continue
                p1 = current[0][0]
                theta1 = current[0][1]
                pt1current = [None, None]
                pt2current = [None, None]
                #If the line is almost horizontal
                if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
                    pt1current[0] = 0
                    pt1current[1] = p1 / np.sin(theta1)
                    pt2current[0] = width
                    pt2current[1] = -pt2current[0] / np.tan(theta1) + p1 / np.sin(theta1)
                #If the line is almost vertical
                else:
                    pt1current[1] = 0
                    pt1current[0] = p1 / np.cos(theta1)
                    pt2current[1] = height
                    pt2current[0] = -pt2current[1] * np.tan(theta1) + p1 / np.cos(theta1)
                #Now to fuse lines
                for pos in lines:
                    if pos[0].all() == current[0].all():
                        continue
                    if abs(pos[0][0] - current[0][0]) < 20 and abs(pos[0][1] - current[0][1]) < np.pi * 10 / 180:
                        p = pos[0][0]
                        theta = pos[0][1]
                        pt1 = [None, None]
                        pt2 = [None, None]
                        # If the line is almost horizontal
                        if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
                            pt1[0] = 0
                            pt1[1] = p / np.sin(theta)
                            pt2[0] = width
                            pt2[1] = -pt2[0] / np.tan(theta) + p / np.sin(theta)
                        # If the line is almost vertical
                        else:
                            pt1[1] = 0
                            pt1[0] = p / np.cos(theta)
                            pt2[1] = height
                            pt2[0] = -pt2[1] * np.tan(theta) + p / np.cos(theta)
                        #If the endpoints are close to each other, merge the lines
                        if (pt1[0] - pt1current[0])**2 + (pt1[1] - pt1current[1])**2 < 64**2 and (pt2[0] - pt2current[0])**2 + (pt2[1] - pt2current[1])**2 < 64**2:
                            current[0][0] = (current[0][0] + pos[0][0]) / 2
                            current[0][1] = (current[0][1] + pos[0][1]) / 2
                            pos[0][0] = None
                            pos[0][1] = None
            #Now to remove the "None" Lines
            lines = list(filter(lambda a : a[0][0] is not None and a[0][1] is not None, lines))
            return lines

#Call the Merge Lines function and store the fused lines
lines = mergeLines(lines, im)

def drawLine(line, img):
    height, width = np.shape(img)
    if line[0][1] != 0:
        m = -1 / np.tan(line[0][1])
        c = line[0][0] / np.sin(line[0][1])
        cv2.line(img, (0, int(c)), (width, int(m * width + c)), 255)
    else:
        cv2.line(img, (line[0][0], 0), (line[0][0], height), 255)
    return img

topedge = [[1000, 1000]]
bottomedge = [[-1000, -1000]]
leftedge = [[1000, 1000]]
leftxintercept = 100000
rightedge = [[-1000, -1000]]
rightxintercept = 0

for i in range(len(lines)):
    current = lines[i][0]
    p = current[0]
    theta = current[1]
    xIntercept = p / np.cos(theta)

    #If the line is nearly vertical
    if theta > np.pi * 80 / 180 and theta < np.pi * 100 / 180:
        if p < topedge[0][0]:
            topedge[0] = current[:]
        if p > bottomedge[0][0]:
            bottomedge[0] = current[:]

    #If the line is nearly horizontal
    if theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
        if xIntercept > rightxintercept:
            rightedge[0] = current[:]
            rightxintercept = xIntercept
        elif xIntercept <= leftxintercept:
            leftedge[0] = current[:]
            leftxintercept = xIntercept

#Drawing the lines
tmpimg= np.copy(outerbox)
tmppp = np.copy(img)
tmppp = drawLine(leftedge, tmppp)
tmppp = drawLine(rightedge, tmppp)
tmppp = drawLine(topedge, tmppp)
tmppp = drawLine(bottomedge, tmppp)

tmpimg = drawLine(leftedge, tmpimg)
tmpimg = drawLine(rightedge, tmpimg)
tmpimg = drawLine(topedge, tmpimg)
tmpimg = drawLine(bottomedge, tmpimg)

leftedge = leftedge[0]
rightedge = rightedge[0]
bottomedge = bottomedge[0]
topedge = topedge[0]

# Calculating two points that lie on each of the four lines
left1 = [None, None]
left2 = [None, None]
right1 = [None, None]
right2 = [None, None]
top1 = [None, None]
top2 = [None, None]
bottom1 = [None, None]
bottom2 = [None, None]

if leftedge[1] != 0:
    left1[0] = 0
    left1[1] = leftedge[0] / np.sin(leftedge[1])
    left2[0] = width
    left2[1] = -left2[0] / np.tan(leftedge[1]) + left1[1]
else:
    left1[1] = 0
    left1[0] = leftedge[0] / np.cos(leftedge[1])
    left2[1] = height
    left2[0] = left1[0] - height * np.tan(leftedge[1])

if rightedge[1] != 0:
    right1[0] = 0
    right1[1] = rightedge[0] / np.sin(rightedge[1])
    right2[0] = width
    right2[1] = -right2[0] / np.tan(rightedge[1]) + right1[1]
else:
    right1[1] = 0
    right1[0] = rightedge[0] / np.cos(rightedge[1])
    right2[1] = height
    right2[0] = right1[0] - height * np.tan(rightedge[1])

bottom1[0] = 0
bottom1[1] = bottomedge[0] / np.sin(bottomedge[1])

bottom2[0] = width
bottom2[1] = -bottom2[0] / np.tan(bottomedge[1]) + bottom1[1]

top1[0] = 0
top1[1] = topedge[0] / np.sin(topedge[1])
top2[0] = width
top2[1] = -top2[0] / np.tan(topedge[1]) + top1[1]

# Next, we find the intersection of these four lines

leftA = left2[1] - left1[1]
leftB = left1[0] - left2[0]
leftC = leftA * left1[0] + leftB * left1[1]

rightA = right2[1] - right1[1]
rightB = right1[0] - right2[0]
rightC = rightA * right1[0] + rightB * right1[1]

topA = top2[1] - top1[1]
topB = top1[0] - top2[0]
topC = topA * top1[0] + topB * top1[1]

bottomA = bottom2[1] - bottom1[1]
bottomB = bottom1[0] - bottom2[0]
bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

# Intersection of left and top

detTopLeft = leftA * topB - leftB * topA

ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)

# Intersection of top and right

detTopRight = rightA * topB - rightB * topA

ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)

# Intersection of right and bottom

detBottomRight = rightA * bottomB - rightB * bottomA

ptBottomRight = ((bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)

# Intersection of bottom and left

detBottomLeft = leftA * bottomB - leftB * bottomA

ptBottomLeft = ((bottomB * leftC - leftB * bottomC) / detBottomLeft,
                        (leftA * bottomC - bottomA * leftC) / detBottomLeft)

# Plotting the found extreme points
cv2.circle(tmppp, (int(ptTopLeft[0]), int(ptTopLeft[1])), 5, 0, -1)
cv2.circle(tmppp, (int(ptTopRight[0]), int(ptTopRight[1])), 5, 0, -1)
cv2.circle(tmppp, (int(ptBottomLeft[0]), int(ptBottomLeft[1])), 5, 0, -1)
cv2.circle(tmppp, (int(ptBottomRight[0]), int(ptBottomRight[1])), 5, 0, -1)

# f.add_subplot(3,3,9)
# plt.imshow(tmppp, cmap="gray")
# plt.title("Houghline Transformation")
# plt.axis('off')

#Finding the maximum length side

leftedgelensq = (ptBottomLeft[0] - ptTopLeft[0])**2 + (ptBottomLeft[1] - ptTopLeft[1])**2
rightedgelensq = (ptBottomRight[0] - ptTopRight[0])**2 + (ptBottomRight[1] - ptTopRight[1])**2
topedgelensq = (ptTopRight[0] - ptTopLeft[0])**2 + (ptTopLeft[1] - ptTopRight[1])**2
bottomedgelensq = (ptBottomRight[0] - ptBottomLeft[0])**2 + (ptBottomLeft[1] - ptBottomRight[1])**2
maxlength = int(max(leftedgelensq, rightedgelensq, bottomedgelensq, topedgelensq)**0.5)

src = [(0, 0)] * 4
dst = [(0, 0)] * 4
src[0] = ptTopLeft[:]
dst[0] = (0, 0)
src[1] = ptTopRight[:]
dst[1] = (maxlength - 1, 0)
src[2] = ptBottomRight[:]
dst[2] = (maxlength - 1, maxlength - 1)
src[3] = ptBottomLeft[:]
dst[3] = (0, maxlength - 1)
src = np.array(src).astype(np.float32)
dst = np.array(dst).astype(np.float32)
extractedgrid = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (maxlength, maxlength))

extractedgrid = cv2.resize(extractedgrid, (252, 252))


f.add_subplot(3,3,9)
plt.imshow(extractedgrid, cmap="gray")
plt.title("Warping the Image")
plt.axis('off')
plt.show()
f2 = plt.figure(figsize=(8,8))
# mask = np.zeros(20,20)

grid = np.copy(extractedgrid)
edge = np.shape(grid)[0]

print('edge: ', edge)
# celledge = edge // 9
celledge = 16
print('celledge: ', celledge)
#Adaptive thresholding the cropped grid and inverting it
bitwise = cv2.bitwise_not(cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))
# grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)
grid = bitwise[4:248, 4:248]
cv2.imwrite(str("grid2.jpg"), grid)


tempgrid = []
for i in range(celledge, edge+1, celledge):
    for j in range(celledge, edge+1, celledge):
        rows = grid[i-celledge:i]
        tempgrid.append([rows[k][j-celledge:j] for k in range(len(rows))])

#Creating the 9X9 grid of images
finalgrid = []
for i in range(0, len(tempgrid)-14, 15):
    finalgrid.append(tempgrid[i:i+15])


#Converting all the cell images to np.array
for i in range(15):
    for j in range(15):
        finalgrid[i][j] = np.array(finalgrid[i][j])

index =0
try:
    for i in range(15):
        for j in range(15):
            os.remove("BoardCells/cell"+str(index)+".jpg")
            index +=1
except:
    pass

index =0
for i in range(15):
    for j in range(15):
        r, c = finalgrid[i][j].shape
        if (r == celledge and c == celledge ):
            cv2.imwrite(str("BoardCells/cell"+str(index)+".jpg"), finalgrid[i][j])
            index +=1

f2.add_subplot(3,3,1)
plt.imshow(grid, cmap="gray")
plt.title("Warping the Image")
plt.axis('off')

plt.show()


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    
myLetters = []
for filename in sorted(os.listdir('BoardCells'), key=numericalSort):
        img = cv2.imread(os.path.join('BoardCells/',filename))
        
        resized_img = resize(img, (16,16)) 
        fd, hog_image = hog(resized_img, pixels_per_cell=(3,3), cells_per_block=(2, 2), visualize=True, multichannel=True)
        
        X = fd
        nX_test = np.delete(X, -1)
        X_test = nX_test.reshape(1,-1)

        pickled_model_svm = pickle.load(open('SVM_Model.pkl', 'rb'))
        predicted_svm = pickled_model_svm.predict(X_test)


        pickled_model_dt = pickle.load(open('DT_Model.pkl', 'rb'))
        predicted_dt = pickled_model_dt.predict(X_test)


        pickled_model_knn = pickle.load(open('KNN_Model.pkl', 'rb'))
        predicted_knn = pickled_model_knn.predict(X_test)

        if ((predicted_svm == predicted_knn) or (predicted_svm == predicted_dt)):
            myLetters.append(predicted_svm)
        elif ((predicted_dt == predicted_knn)):
            myLetters.append(predicted_dt)
        else:
            myLetters.append(predicted_dt)

index = 0
letters = np.zeros(shape=(15, 15), dtype=myLetters[0].dtype)

for i in range(15):
    for j in range(15):
        letters[i,j] = myLetters[index]
        index = index + 1


root = Tk()
			
root.geometry('1000x500')	
root.title("Scramble Puzzle ")
Label(root, text ="Scramble Puzzle Solver", font=150).pack()

class WordSearch(object):
      
      def __init__(self):
        self.word = StringVar()

      def ViewWords(self, letters):
                  
            dframe = pd.DataFrame(letters)

            txt = Text(root) 
            txt.pack() 

            class PrintToTXT(object): 
                  def write(self, s):     
                        txt.insert(END, s)
                  def flush(self):
                        pass
            sys.stdout = PrintToTXT() 

            print ('Characters found in the image are:') 

            print (dframe)

            Label(root, text="Enter Word to Search:").place(x=100, y=500)
            w = Entry(root, textvariable=self.word)
            w.place(x=250, y=500)

            
            submit = Button(root, text = 'Submit', command = lambda: self.FindWords(letters))
            submit.place(x=100,y=550) 

            
      def FindWords(self, letters):
            
            try: 
                  word = self.word.get()
                  self.find_word(letters, word)
            except:
                  print('Word not found')
            
      def find_word (self, wordsearch, word='Abcd'):
            """Trys to find word in wordsearch and prints result"""
            # Store first character positions in array
            print('word is: ', word)
            start_pos = []
            
            first_char = word[0]

            for i in range(0, len(wordsearch)):
                  for j in range(0, len(wordsearch[i])):
                        if (wordsearch[i][j] == first_char):
                              start_pos.append([i,j])
            # Check all starting positions for word
            for p in start_pos:
                  if self.check_start(wordsearch, word, p):
                        # Word found
                        return
            # Word not found
            print('Word Not Found')

      def check_start (self, wordsearch, word, start_pos):
            """Checks if the word starts at the startPos. Returns True if word found"""
            directions = [[-1,1], [0,1], [1,1], [-1,0], [1,0], [-1,-1], [0,-1], [1,-1]]
            # Iterate through all directions and check each for the word
            for d in directions:
                  if (self.check_dir(wordsearch, word, start_pos, d)):
                        return True

      def check_dir (self, wordsearch, word, start_pos, dir):
            """Checks if the word is in a direction dir from the start_pos position in the wordsearch. Returns True and prints result if word found"""
            found_chars = [word[0]] # Characters found in direction. Already found the first character
            current_pos = start_pos # Position we are looking at
            pos = [start_pos] # Positions we have looked at
            while (self.chars_match(found_chars, word)):
                  if (len(found_chars) == len(word)):
                        # If found all characters and all characters found are correct, then word has been found
                        print('')
                        print('Word Found')
                        print('')
                        # Draw wordsearch on command line. Display found characters and '-' everywhere else
                        for x in range(0, len(wordsearch)):
                              line = ""
                              for y in range(0, len(wordsearch[x])):
                                    is_pos = False
                                    for z in pos:
                                          if (z[0] == x) and (z[1] == y):
                                                is_pos = True
                                    if (is_pos):
                                          line = line + " " + wordsearch[x][y]
                                    else:
                                          line = line + " -"
                              print(line)
                        print('')
                        return True;
                  # Have not found enough letters so look at the next one
                  current_pos = [current_pos[0] + dir[0], current_pos[1] + dir[1]]
                  pos.append(current_pos)
                  if (self.is_valid_index(wordsearch, current_pos[0], current_pos[1])):
                        found_chars.append(wordsearch[current_pos[0]][current_pos[1]])
                  else:
                        # Reached edge of wordsearch and not found word
                        return

      def chars_match (self, found, word):
            """Checks if the leters found are the start of the word we are looking for"""
            index = 0
            for i in found:
                  if (i != word[index]):
                        return False
                  index += 1
            return True

      def is_valid_index (self, wordsearch, line_num, col_num):
            """Checks if the provided line number and column number are valid"""
            if ((line_num >= 0) and (line_num < len(wordsearch))):
                  if ((col_num >= 0) and (col_num < len(wordsearch[line_num]))):
                        return True
            return False


a = WordSearch()
a.ViewWords(letters)

root.mainloop()