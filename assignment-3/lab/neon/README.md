# ARM Neon Demo

The starter code in this folder loads a vector from file and sums each element into a single result.  Examine `main.c` and use it as your starting point.

Your goal for this demo is use the ARM Neon Intrinsics to do a vector addition of 4 elements into one.  You will later leverage this for the ARM Neon block matrix multiply in the homework.  Use the resources in the top-level readme as a guide.

1. Load the current 4 elements into a quadword
2. Sum all four elements of the quadword
3. Extract the value from the quad word
4. Increment by 4 instead of by 1

Questions to consider:

* How do we handle when there is not at least four elements remaining?
* What is happening at the hardware level that may make this more efficient?
