#include "huffman.h"
#include <vector>
#include <array>
#include <iostream>
#include <queue>
#include <string>
#include <cmath>
#include <map>

using namespace std;

struct Node
{
	unsigned char letter;
	unsigned int freq;
	int index;
	int leftIndex;
	int rightIndex;
};

struct CompareNode
{
	bool operator()(const Node &left, const Node &right) const
	{
		if (left.freq == right.freq)
		{
			return left.index < right.index;
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
int huffman_encode(const unsigned char *bufin,
				   unsigned int bufinlen,
				   unsigned char **pbufout,
				   unsigned int *pbufoutlen)
{
	if (pbufout == nullptr || pbufoutlen == nullptr)
	{
		return -1;
	}

	*pbufout = nullptr;
	*pbufoutlen = 0;

	if (bufin == nullptr || bufinlen == 0)
	{
		return 0;
	}

	vector<Node> frequencies = histogram(bufin, bufinlen);
	const std::vector<Node> originalFrequencies = frequencies;

	priority_queue<Node, vector<Node>, CompareNode> minHeap(CompareNode{}, frequencies);

	// DEBUG
	auto debugHeap = minHeap;

	while (!debugHeap.empty())
	{
		Node node = debugHeap.top();
		debugHeap.pop();

		cout << static_cast<char>(node.letter)
			 << ": "
			 << node.freq
			 << endl;
	} // end debug

	// Build Huffman tree
	while (minHeap.size() > 1)
	{
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
	cout << "\nHuffman Tree Codes:\n";
	int leafCount = originalFrequencies.size();
	if (leafCount == 1)
	{
		cout << originalFrequencies[0].letter
			 << " = 0"
			 << endl;
	}
	else
	{
		printTree(frequencies, rootIndex, "");
	} // end debug

	// Generate symbol codewords
	std::map<unsigned char, string> codes;
	generateCodes(frequencies, rootIndex, "", codes);

	if (leafCount == 1)
	{
		codes[frequencies[0].letter] = "0";
	}

	// Count encoded bits for each symbol
	// Count encoded bits using only histogram leaf nodes
	unsigned int encodedBitCount = 0;

	for (unsigned int i = 0; i < leafCount; ++i)
	{
		encodedBitCount +=
			frequencies[i].freq *
			static_cast<unsigned int>(
				codes[frequencies[i].letter].size());
	}

	// convert to total bits to bytes
	unsigned int encodedByteCount = static_cast<unsigned int>(std::ceil(encodedBitCount / 8.0));
	// unsigned int is 4bytes, each symbol is 1 byte (char) so 5bytes per entry
	// plus 4bytes
	// plus 4bytes

	unsigned int headerSize =
		sizeof(unsigned int) +										// input length
		sizeof(unsigned int) +										// number of histogram entries
		leafCount * (sizeof(unsigned char) + sizeof(unsigned int)); // each entry

	*pbufoutlen = headerSize + encodedByteCount;
	*pbufout = new unsigned char[*pbufoutlen]{};

	// Store original input length
	writeUint32(*pbufout, 0, bufinlen);
	// Next 4 bytes: number of histogram entries
	writeUint32(*pbufout, sizeof(unsigned int), leafCount);
	// Store frequency table
	unsigned int outputPosition = sizeof(unsigned int) + sizeof(unsigned int);

	for (unsigned int i = 0; i < leafCount; ++i)
	{
		// 1 byte: symbol
		(*pbufout)[outputPosition] = originalFrequencies[i].letter;
		outputPosition += sizeof(unsigned char);
		// 4 bytes: frequency
		writeUint32(*pbufout, outputPosition, originalFrequencies[i].freq);
		outputPosition += sizeof(unsigned int);
	}

	// Encode input symbols using the code words
	unsigned int bitPosition = 0;
	for (unsigned int i = 0; i < bufinlen; ++i)
	{
		const string &code = codes[bufin[i]];
		for (char bit : code)
		{
			unsigned int byteIndex = headerSize + (bitPosition / 8);
			unsigned int bitIndex = 7 - (bitPosition % 8);

			if (bit == '1')
			{
				(*pbufout)[byteIndex] |= 1 << bitIndex; // turn bit on
			}

			++bitPosition;
		}
	}

	return 0;
}

void generateCodes(const vector<Node> &nodes, int nodeIndex, const string &currentCode, std::map<unsigned char, string> &codes)
{
	const Node &node = nodes[nodeIndex];

	// Leaf node base case
	if (node.leftIndex == -1 && node.rightIndex == -1)
	{
		codes[node.letter] = currentCode;
		return;
	}

	// Add a zero for left child
	if (node.leftIndex != -1)
	{
		generateCodes(nodes, node.leftIndex, currentCode + "0", codes);
	}

	// Add a 1 for right child
	if (node.rightIndex != -1)
	{
		generateCodes(nodes, node.rightIndex, currentCode + "1", codes);
	}
}

void printTree(const vector<Node> &nodes, int index, string code)
{
	const Node &node = nodes[index];

	bool isLeaf =
		node.leftIndex == -1 &&
		node.rightIndex == -1;

	if (isLeaf)
	{
		cout << node.letter
			 << " = "
			 << code
			 << endl;

		return;
	}

	printTree(nodes, node.leftIndex, code + "0");
	printTree(nodes, node.rightIndex, code + "1");
}

vector<Node> histogram(const unsigned char *bufin, unsigned int bufinlen)
{
	vector<Node> result;

	if (bufin == nullptr || bufinlen == 0)
	{
		return result;
	}

	// Max unsigned char values is 255.
	// array<unsigned int, 256> counts{};
	map<unsigned char, unsigned int> counts;
	// calculate frequency from input symbols
	for (unsigned int i = 0; i < bufinlen; ++i)
	{
		++counts[bufin[i]];
	}

	// create nodes
	for (const auto &entry : counts)
	{
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
// transform unsigned int (4 bytes) to char (1 byte)
// 32bits to 8bits
// so need to partition value into 4 segments of 8bits
void writeUint32(unsigned char *buffer, unsigned int position, unsigned int value)
{
	buffer[position] = static_cast<unsigned char>(value & 0xFF);			 // first 1byte
	buffer[position + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);	 // 2nd byte
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
int huffman_decode(const unsigned char *bufin,
				   unsigned int bufinlen,
				   unsigned char **pbufout,
				   unsigned int *pbufoutlen)
{

	unsigned int originalLength = readUint32(bufin, 0);
	unsigned int leafCount = readUint32(bufin, sizeof(unsigned int));
	unsigned int inputPosition = sizeof(unsigned int) + sizeof(unsigned int);

	*pbufoutlen = originalLength;
	*pbufout = new unsigned char[originalLength];

	vector<Node> frequencies;

	for (unsigned int i = 0; i < leafCount; ++i)
	{
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

		frequencies.push_back(node);
	}

	priority_queue<Node, vector<Node>, CompareNode> minHeap(CompareNode{}, frequencies);

	// Rebuild the Huffman tree.
	while (minHeap.size() > 1)
	{
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

	// Handle an input containing only one unique symbol.
	if (frequencies[rootIndex].leftIndex == -1 &&
		frequencies[rootIndex].rightIndex == -1)
	{
		for (unsigned int i = 0; i < originalLength; ++i)
		{
			(*pbufout)[i] =
				frequencies[rootIndex].letter;
		}

		return 0;
	}

	unsigned int headerSize = sizeof(unsigned int) + sizeof(unsigned int) + leafCount * (sizeof(unsigned char) + sizeof(unsigned int));
	unsigned int bitPosition = 0;
	unsigned int outputPosition = 0;
	int currentIndex = rootIndex;

	// Decode bits until the original number of bytes is restored.
	while (outputPosition < originalLength)
	{
		unsigned int byteIndex = headerSize + (bitPosition / 8);
		unsigned int bitIndex = 7 - (bitPosition % 8);
		unsigned char bit = (bufin[byteIndex] >> bitIndex) & 1;

		if (bit == 0)
		{
			currentIndex = frequencies[currentIndex].leftIndex;
		}
		else
		{
			currentIndex = frequencies[currentIndex].rightIndex;
		}

		++bitPosition;

		const Node &node = frequencies[currentIndex];

		// Reached a leaf node.
		if (node.leftIndex == -1 && node.rightIndex == -1)
		{
			(*pbufout)[outputPosition] = node.letter;
			++outputPosition;
			currentIndex = rootIndex;
		}
	}

	return 0;
}

unsigned int readUint32(const unsigned char *buffer, unsigned int position)
{
	unsigned int value = 0;

	value |= buffer[position];
	value |= static_cast<unsigned int>(buffer[position + 1]) << 8;
	value |= static_cast<unsigned int>(buffer[position + 2]) << 16;
	value |= static_cast<unsigned int>(buffer[position + 3]) << 24;

	return value;
}