# Loop Unrolling Demo

For this demo, copy the starter code from the `loop` folder.  You will use that as a starting point.

Your goal for this demo is to have four add instructions in the loop.  You will later leverage this for doing the ARM Neon Intrinsics.

1. Manually duplicate the loop body
2. Increment by 4 instead of by 1

Questions to consider:

* How do we handle when there is not at least four elements remaining?
- Answer: add a condition if statement to verify index is valid.
* What is happening at the instruction level that may make this more efficient?
- Well a loop is is indexing a contigous data 4bytes at a time. In each loop it then takes that element value and add to the runnning sum value
- Incrementing by 4 is indexing the array by 32bytes each time instead of 4. So it's takes less iterations in the loop. Still runs in O(n) but the actual constant is most likely 1/4(n).


[1] J. L. Hennessy and D. A. Patterson, Computer Architecture: A Quantitative Approach, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.
