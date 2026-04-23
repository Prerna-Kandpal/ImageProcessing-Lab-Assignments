import cv2
import numpy as np
from collections import Counter
import heapq

# -------------------------------
# Read Image (Grayscale)
# -------------------------------
img = cv2.imread('C:/Users/91986/Desktop/color_segmentation.jpg', 0)

if img is None:
    print("Error: Image not found!")
    exit()

# -------------------------------
# Flatten Image
# -------------------------------
flat = img.flatten()

# -------------------------------
# Run Length Encoding (RLE)
# -------------------------------
def rle_encode(data):
    encoding = []
    prev = data[0]
    count = 1

    for i in range(1, len(data)):
        if data[i] == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = data[i]
            count = 1

    encoding.append((prev, count))
    return encoding

rle_data = rle_encode(flat)

# -------------------------------
# Huffman Coding
# -------------------------------
class Node:
    def __init__(self, freq, pixel, left=None, right=None):
        self.freq = freq
        self.pixel = pixel
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = Counter(data)
    heap = []

    for pixel, f in freq.items():
        heapq.heappush(heap, Node(f, pixel))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = Node(left.freq + right.freq, None, left, right)
        heapq.heappush(heap, merged)

    return heap[0]

def build_codes(node, current_code="", codes={}):
    if node is None:
        return

    if node.pixel is not None:
        codes[node.pixel] = current_code
        return

    build_codes(node.left, current_code + "0", codes)
    build_codes(node.right, current_code + "1", codes)

huffman_tree = build_huffman_tree(flat)
huffman_codes = {}
build_codes(huffman_tree, "", huffman_codes)

# Encode using Huffman
encoded_data = ''.join([huffman_codes[pixel] for pixel in flat])

# -------------------------------
# Compression Ratio
# -------------------------------
original_size = len(flat) * 8   # 8 bits per pixel
compressed_size = len(encoded_data)

compression_ratio = original_size / compressed_size

# -------------------------------
# Output
# -------------------------------
print("Original Size (bits):", original_size)
print("Compressed Size (bits):", compressed_size)
print("Compression Ratio:", round(compression_ratio, 2))

print("\nRLE Length:", len(rle_data))
print("Huffman Codes (sample):", list(huffman_codes.items())[:10])