# Loop Unrolling Demo

For this demo, copy the starter code from the `loop` folder.  You will use that as a starting point.

Your goal for this demo is to have four add instructions in the loop.  You will later leverage this for doing the ARM Neon Intrinsics.

1. Manually duplicate the loop body
2. Increment by 4 instead of by 1

Questions to consider:

* How do we handle when there is not at least four elements remaining?
* What is happening at the instruction level that may make this more efficient?