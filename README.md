# Personalised Page Rank

Page Rank works with an input of an occurence adjacency list. First, convert the plain-text adjacency list representation into Hadoop Writable records:

```
$ hadoop jar target/assignments-1.0.jar \
   ca.uwaterloo.cs451.a4.BuildPersonalizedPageRankRecords \
   -input data/p2p-Gnutella08-adj.txt -output cs451-bigdatateach-a4-Gnutella-PageRankRecords \
   -numNodes 6301 -sources 367,249,145
```
The -sources option specifies the source nodes for the personalized PageRank computations. You can expect the option value to be in the form of a comma-separated list, and that all node ids actually exist in the graph. The list of source nodes may be arbitrarily long, but for practical purposes we won't test your code with more than a few.
Since we're running three personalized PageRank computations in parallel, each PageRank node is going to hold an array of three values, the personalized PageRank values with respect to the first source, second source, and third source. You can expect the array positions to correspond exactly to the position of the node id in the source string.

Next, partition the graph (hash partitioning) and get ready to iterate:

```
$ hadoop fs -mkdir cs451-bigdatateach-a4-Gnutella-PageRank

$ hadoop jar target/assignments-1.0.jar \
   ca.uwaterloo.cs451.a4.PartitionGraph \
   -input cs451-bigdatateach-a4-Gnutella-PageRankRecords \
   -output cs451-bigdatateach-a4-Gnutella-PageRank/iter0000 -numPartitions 5 -numNodes 6301
```

After setting everything up, iterate multi-source personalized PageRank:

```
$ hadoop jar target/assignments-1.0.jar \
   ca.uwaterloo.cs451.a4.RunPersonalizedPageRankBasic \
   -base cs451-bigdatateach-a4-Gnutella-PageRank -numNodes 6301 -start 0 -end 20 -sources 367,249,145
```

Note that the sources are passed in from the command-line again. Here, we're running twenty iterations.

Finally, run a program to extract the top ten personalized PageRank values, with respect to each source.

```
$ hadoop jar target/assignments-1.0.jar \
   ca.uwaterloo.cs451.a4.ExtractTopPersonalizedPageRankNodes \
   -input cs451-bigdatateach-a4-Gnutella-PageRank/iter0020 -output cs451-bigdatateach-a4-Gnutella-PageRank-top10 \
   -top 10 -sources 367,249,145
```


### More on how Personalised Page Rank works

Regular Page Rank works by working through a connected graph and adjusting probability weights where the probability is adjusted through every iteration.
Usually to offset the effects of dead ends and cyclical links, random jumps are set in place and dead ends are filled up. However personalised page rank is different in several ways:
- There is the notion of a source node, which is what we're computing the personalization with respect to.
- When initializing PageRank, instead of a uniform distribution across all nodes, the source node gets a mass of one and every other node gets a mass of zero.
- Whenever the model makes a random jump, the random jump is always back to the source node; this is unlike in ordinary PageRank, where there is an equal probability of jumping to any node.
- All mass lost in the dangling nodes are put back into the source node; this is unlike in ordinary PageRank, where the missing mass is evenly distributed across all nodes.

Some publications on personalised page rank: 
 - Daniel Fogaras, Balazs Racz, Karoly Csalogany, and Tamas Sarlos. (2005) : https://projecteuclid.org/download/pdf_1/euclid.im/1150474886
