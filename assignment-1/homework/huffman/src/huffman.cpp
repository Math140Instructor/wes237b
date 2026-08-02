#include "huffman.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <vector>

using namespace std;

struct Node {
  unsigned char letter;
  unsigned int freq;
  int index;
  int leftIndex;
  int rightIndex;
};

struct CompareNode {
  bool operator()(const Node &left, const Node &right) const {
    if (left.freq == right.freq) {
      return left.index > right.index;
    }
    return left.freq > right.freq;
  }
};

// Helpers
vector<Node> histogram(const unsigned char *bufin, unsigned int bufinlen);
void printTree(const vector<Node> &nodes, int index, string code);
void generateCodes(const vector<Node> &nodes, int nodeIndex, const string &currentCode, std::map<unsigned char, string> &codes);
void writeUint32(unsigned char *buffer, unsigned int position, unsigned int value);
unsigned int readUint32(const unsigned char *buffer, unsigned int position);

/**
 * Create histogram
 * Create min-heap
 * Build Huffman tree
 * Generate symbol codewords
 * Encode the input
 * Store enough tree/frequency information for decoding
 * Allocate the output buffer
 * Set code and code_size
 **/
int huffman_encode(const unsigned char *bufin, unsigned int bufinlen, unsigned char **pbufout, unsigned int *pbufoutlen) {
  if (pbufout == nullptr || pbufoutlen == nullptr) {
    return -1;
  }

  *pbufout = nullptr;
  *pbufoutlen = 0;

  if (bufin == nullptr || bufinlen == 0) {
    return 0;
  }

  vector<Node> frequencies = histogram(bufin, bufinlen);
  const std::vector<Node> originalFrequencies = frequencies;

  priority_queue<Node, vector<Node>, CompareNode> minHeap(CompareNode{}, frequencies);

  // DEBUG
  //   auto debugHeap = minHeap;
  //   while (!debugHeap.empty()) {
  //     Node node = debugHeap.top();
  //     debugHeap.pop();

  //     cout << static_cast<char>(node.letter) << ": " << node.freq << endl;
  //   } // end debug

  // Build Huffman tree
  while (minHeap.size() > 1) {
    Node left = minHeap.top();
    minHeap.pop();

    Node right = minHeap.top();
    minHeap.pop();

    Node parent;
    parent.letter = '\0';
    parent.freq = left.freq + right.freq;
    parent.leftIndex = left.index;
    parent.rightIndex = right.index;
    parent.index = static_cast<int>(frequencies.size());

    frequencies.push_back(parent);
    minHeap.push(parent);
  }

  int rootIndex = minHeap.top().index;

  /**
   * DEBUG
   */
  // cout << "\nHuffman Tree Codes:\n";
  unsigned int leafCount = static_cast<unsigned int>(originalFrequencies.size());

  //   if (leafCount == 1) {
  //     cout << originalFrequencies[0].letter << " = 0" << endl;
  //   } else {
  //     printTree(frequencies, rootIndex, "");
  //   } // end debug

  // Generate symbol codewords
  std::map<unsigned char, string> codes;
  generateCodes(frequencies, rootIndex, "", codes);

  if (leafCount == 1) {
    codes[frequencies[0].letter] = "0";
  }

  // Count encoded bits for each symbol
  // Count encoded bits using only histogram leaf nodes
  unsigned int encodedBitCount = 0;

  for (unsigned int i = 0; i < leafCount; ++i) {
    encodedBitCount += frequencies[i].freq * static_cast<unsigned int>(codes[frequencies[i].letter].size());
  }

  // convert to total bits to bytes
  unsigned int encodedByteCount = static_cast<unsigned int>(std::ceil(encodedBitCount / 8.0));
  // unsigned int is 4bytes, each symbol is 1 byte (char) so 5bytes per entry
  // plus 4bytes
  // plus 4bytes

  unsigned int headerSize = sizeof(unsigned int) +                                      // input length
                            sizeof(unsigned int) +                                      // number of histogram entries
                            leafCount * (sizeof(unsigned char) + sizeof(unsigned int)); // each entry

  *pbufoutlen = headerSize + encodedByteCount;
  //*pbufout = new unsigned char[*pbufoutlen]{};
  *pbufout = static_cast<unsigned char *>(calloc(*pbufoutlen, sizeof(unsigned char)));

  if (*pbufout == nullptr) {
    *pbufoutlen = 0;
    return -1;
  }

  // Store original input length
  writeUint32(*pbufout, 0, bufinlen);
  // Next 4 bytes: number of histogram entries
  writeUint32(*pbufout, sizeof(unsigned int), leafCount);
  // Store histogram in the next 4 bytes hence 4+4.
  unsigned int outputPosition = sizeof(unsigned int) + sizeof(unsigned int);
  for (unsigned int i = 0; i < leafCount; ++i) {
    // 1 byte: symbol
    (*pbufout)[outputPosition] = originalFrequencies[i].letter;
    outputPosition += sizeof(unsigned char);
    // 4 bytes: frequency
    writeUint32(*pbufout, outputPosition, originalFrequencies[i].freq);
    outputPosition += sizeof(unsigned int);
  }

  // Encode input symbols using the code words
  unsigned int bitPosition = 0;
  for (unsigned int i = 0; i < bufinlen; ++i) {
    const string &code = codes[bufin[i]];
    for (char bit : code) {
      unsigned int byteIndex = headerSize + (bitPosition / 8);
      unsigned int bitIndex = 7 - (bitPosition % 8);

      if (bit == '1') {
        (*pbufout)[byteIndex] |= 1 << bitIndex; // turn bit on
      }

      ++bitPosition;
    }
  }

  return 0;
}

void generateCodes(const vector<Node> &nodes, int nodeIndex, const string &currentCode, std::map<unsigned char, string> &codes) {
  const Node &node = nodes[nodeIndex];

  // Leaf node base case
  if (node.leftIndex == -1 && node.rightIndex == -1) {
    codes[node.letter] = currentCode;
    return;
  }

  // Add a zero for left child
  if (node.leftIndex != -1) {
    generateCodes(nodes, node.leftIndex, currentCode + "0", codes);
  }

  // Add a 1 for right child
  if (node.rightIndex != -1) {
    generateCodes(nodes, node.rightIndex, currentCode + "1", codes);
  }
}

/**
 * Debug: visually show tree via console outputs.
 */
void printTree(const vector<Node> &nodes, int index, string code) {
  const Node &node = nodes[index];

  bool isLeaf = node.leftIndex == -1 && node.rightIndex == -1;

  if (isLeaf) {
    cout << node.letter << " = " << code << endl;
    return;
  }

  printTree(nodes, node.leftIndex, code + "0");
  printTree(nodes, node.rightIndex, code + "1");
}

/**
 * Calculate frequencies. No sort.
 */
vector<Node> histogram(const unsigned char *bufin, unsigned int bufinlen) {
  vector<Node> result;

  if (bufin == nullptr || bufinlen == 0) {
    return result;
  }
  map<unsigned char, unsigned int> counts;
  // calculate frequency from input symbols
  for (unsigned int i = 0; i < bufinlen; ++i) {
    ++counts[bufin[i]];
  }

  // create nodes to encapsulate data
  for (const auto &entry : counts) {
    Node node;
    node.letter = entry.first;
    node.freq = entry.second;
    node.index = static_cast<int>(result.size());
    node.leftIndex = -1;
    node.rightIndex = -1;
    result.push_back(node);
  }
  return result; // sorted histogram
}
// Store a 32bit unsigned integer in a byte buffer.
// An unsigned int uses 4 bytes, while each buffer element stores 1 byte.
// So split the 32-bit value into four 8bit bytes and store them in little endian order.
void writeUint32(unsigned char *buffer, unsigned int position, unsigned int value) {
  buffer[position] = static_cast<unsigned char>(value & 0xFF);             // first 1byte
  buffer[position + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);  // 2nd byte
  buffer[position + 2] = static_cast<unsigned char>((value >> 16) & 0xFF); // 3rd byte
  buffer[position + 3] = static_cast<unsigned char>((value >> 24) & 0xFF); // 4th byte
}

/**
 * Read the stored tree/frequency information
 * Reconstruct the Huffman tree
 * Decode the encoded bits
 * Allocate the decoded buffer
 * Set decode and decode_size
 **/
int huffman_decode(const unsigned char *bufin, unsigned int bufinlen, unsigned char **pbufout, unsigned int *pbufoutlen) {

  if (pbufout == nullptr || pbufoutlen == nullptr) {
    return -1;
  }

  *pbufout = nullptr;
  *pbufoutlen = 0;

  if (bufin == nullptr || bufinlen == 0) {
    return 0;
  }

  if (bufinlen < 8) {
    return -1;
  }

  unsigned int originalLength = readUint32(bufin, 0);                       // Number of bytes in the original uncompressed input. Example: "hello world" contains 11 characters at 1 byte each thus 11 bytes.
  unsigned int leafCount = readUint32(bufin, sizeof(unsigned int));         // histogram number of entries.
  unsigned int inputPosition = sizeof(unsigned int) + sizeof(unsigned int); // start after reading the first two ints (8-byte offset, hence 4 + 4)

  // A byte-based Huffman tree can contain at most 256 unique symbols.
  if (leafCount == 0 || leafCount > 256) {
    return -1;
  }

  // Verify that the complete histogram header is present before reading it.
  unsigned int headerSize = sizeof(unsigned int) + sizeof(unsigned int) + leafCount * (sizeof(unsigned char) + sizeof(unsigned int));

  if (headerSize > bufinlen) {
    return -1;
  }

  *pbufoutlen = originalLength;
  //*pbufout = new unsigned char[originalLength]; // original string length
  *pbufout = static_cast<unsigned char *>(calloc(originalLength + 1, sizeof(unsigned char)));

  if (*pbufout == nullptr) {
    *pbufoutlen = 0;
    return -1;
  }
  // recreate Huffman tree
  // Need to read the histogram mapping
  vector<Node> frequencies;
  for (unsigned int i = 0; i < leafCount; ++i) {

    unsigned char symbol = bufin[inputPosition];
    inputPosition += sizeof(unsigned char);

    unsigned int frequency = readUint32(bufin, inputPosition);
    inputPosition += sizeof(unsigned int);

    Node node;
    node.letter = symbol;
    node.freq = frequency;
    node.index = static_cast<int>(frequencies.size());
    node.leftIndex = -1;
    node.rightIndex = -1;

    frequencies.emplace_back(node);
  }

  // sort the histogram
  priority_queue<Node, vector<Node>, CompareNode> minHeap(CompareNode{}, frequencies);

  // Rebuild the Huffman tree.
  while (minHeap.size() > 1) {
    Node left = minHeap.top();
    minHeap.pop();

    Node right = minHeap.top();
    minHeap.pop();

    Node parent;
    parent.letter = '\0';
    parent.freq = left.freq + right.freq;
    parent.leftIndex = left.index;
    parent.rightIndex = right.index;
    parent.index = static_cast<int>(frequencies.size());
    frequencies.push_back(parent);
    minHeap.push(parent);
  }

  if (minHeap.empty()) {
    free(*pbufout);
    *pbufout = nullptr;
    *pbufoutlen = 0;
    return -1;
  }

  int rootIndex = minHeap.top().index;

  // Handle only one unique symbol.
  if (frequencies[rootIndex].leftIndex == -1 && frequencies[rootIndex].rightIndex == -1) {
    for (unsigned int i = 0; i < originalLength; ++i) {
      (*pbufout)[i] = frequencies[rootIndex].letter;
    }
    return 0;
  }

  // The encoded bitstream starts immediately after the header.
  unsigned int bitPosition = 0;
  unsigned int outputPosition = 0;
  int currentIndex = rootIndex;

  // Decode bits until the original number of bytes has been reconstructed.
  // Use each encoded bit to traverse the Huffman tree.
  while (outputPosition < originalLength) {
    // Locate the byte containing the current encoded bit.
    // headerSize skips past the stored header information.
    unsigned int byteIndex = headerSize + (bitPosition / 8);
    // Locate the current bit within that byte.
    // Bits are read from the most significant bit to the least significant bit:
    // bit 7, 6, 5 =,..., 0.
    unsigned int bitIndex = 7 - (bitPosition % 8);
    if (byteIndex >= bufinlen) {
      free(*pbufout);
      *pbufout = nullptr;
      *pbufoutlen = 0;
      return -1;
    }

    // keep only that bit.
    unsigned char bit = (bufin[byteIndex] >> bitIndex) & 1;

    if (bit == 0) {
      currentIndex = frequencies[currentIndex].leftIndex;
    } else {
      currentIndex = frequencies[currentIndex].rightIndex;
    }

    // Move to the next encoded bit.
    ++bitPosition;

    if (currentIndex < 0 || static_cast<unsigned int>(currentIndex) >= frequencies.size()) {
      free(*pbufout);
      *pbufout = nullptr;
      *pbufoutlen = 0;
      return -1;
    }

    const Node &node = frequencies[currentIndex];

    // Leaf node means we have decoded the symbol!
    if (node.leftIndex == -1 && node.rightIndex == -1) {
      // Store the decoded symbol in the output buffer.
      (*pbufout)[outputPosition] = node.letter;
      ++outputPosition;
      // Return to the root to decode the next symbol.
      currentIndex = rootIndex;
    }
  }

  return 0;
}

unsigned int readUint32(const unsigned char *buffer, unsigned int position) {
  unsigned int value = 0;

  value |= buffer[position];
  value |= static_cast<unsigned int>(buffer[position + 1]) << 8;
  value |= static_cast<unsigned int>(buffer[position + 2]) << 16;
  value |= static_cast<unsigned int>(buffer[position + 3]) << 24;

  return value;
}