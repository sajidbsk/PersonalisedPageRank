/**
 * Bespin: reference implementations of "big data" algorithms
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ca.uwaterloo.cs451.a4;

import com.google.common.base.Preconditions;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import tl.lin.data.array.ArrayListOfIntsWritable;
import tl.lin.data.array.ArrayListOfFloatsWritable;
import tl.lin.data.map.HMapIF;
import tl.lin.data.map.MapIF;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Iterator;

/**
 * <p>
 * Main driver program for running the basic (non-Schimmy) implementation of
 * PageRank.
 * </p>
 *
 * <p>
 * The starting and ending iterations will correspond to paths
 * <code>/base/path/iterXXXX</code> and <code>/base/path/iterYYYY</code>. As a
 * example, if you specify 0 and 10 as the starting and ending iterations, the
 * driver program will start with the graph structure stored at
 * <code>/base/path/iter0000</code>; final results will be stored at
 * <code>/base/path/iter0010</code>.
 * </p>
 *
 * @see RunPageRankSchimmy
 * @author Jimmy Lin
 * @author Michael Schatz
 */
public class RunPersonalizedPageRankBasic extends Configured implements Tool {
  private static final Logger LOG = Logger.getLogger(RunPersonalizedPageRankBasic.class);

  private static enum PageRank {
    nodes, edges, massMessages, massMessagesSaved, massMessagesReceived, missingStructure
  };

  // Mapper, no in-mapper combining.
  private static class MapClass extends
      Mapper<IntWritable, PageRankNode, IntWritable, PageRankNode> {

    // The neighbor to which we're sending messages.
    private static final IntWritable neighbor = new IntWritable();

    // Contents of the messages: partial PageRank mass.
    private static final PageRankNode intermediateMass = new PageRankNode();

    // For passing along node structure.
    private static final PageRankNode intermediateStructure = new PageRankNode();

    private static String[] sources;

    @Override
    public void setup(Context context) throws IOException {
      sources = context.getConfiguration().getStrings(SOURCES);
    }

    @Override
    public void map(IntWritable nid, PageRankNode node, Context context)
        throws IOException, InterruptedException {
      // Pass along node structure.
      intermediateStructure.setNodeId(node.getNodeId());
      intermediateStructure.setType(PageRankNode.Type.Structure);
      intermediateStructure.setAdjacencyList(node.getAdjacencyList());

      context.write(nid, intermediateStructure);

      int massMessages = 0;

      // Distribute PageRank mass to neighbors (along outgoing edges).
      if (node.getAdjacencyList().size() > 0) {
        // Each neighbor gets an equal share of PageRank mass.
        ArrayListOfIntsWritable list = node.getAdjacencyList();
        ArrayListOfFloatsWritable mass = new ArrayListOfFloatsWritable();

        for (int i = 0; i < sources.length; i++) {
          mass.add(node.getPageRank().get(i) - (float) StrictMath.log(list.size()));
        }

        context.getCounter(PageRank.edges).increment(list.size());

        // Iterate over neighbors.
        for (int i = 0; i < list.size(); i++) {
          neighbor.set(list.get(i));
          intermediateMass.setNodeId(list.get(i));
          intermediateMass.setType(PageRankNode.Type.Mass);
          intermediateMass.setPageRank(mass);

          // Emit messages with PageRank mass to neighbors.
          context.write(neighbor, intermediateMass);
          massMessages++;
        }
      }

      // Bookkeeping.
      context.getCounter(PageRank.nodes).increment(1);
      context.getCounter(PageRank.massMessages).increment(massMessages);
    }
  }

  // Combiner: sums partial PageRank contributions and passes node structure along.
  private static class CombineClass extends
      Reducer<IntWritable, PageRankNode, IntWritable, PageRankNode> {
    private static final PageRankNode intermediateMass = new PageRankNode();
    private static String[] sources;

    @Override
    public void reduce(IntWritable nid, Iterable<PageRankNode> values, Context context)
        throws IOException, InterruptedException {
      int massMessages = 0;

      String sourceList = context.getConfiguration().get("SOURCES", "");
      sources = sourceList.split(",");

      // Remember, PageRank mass is stored as a log prob.
      ArrayListOfFloatsWritable mass = new ArrayListOfFloatsWritable();

      for (int i = 0; i < sources.length; i++) {
        mass.add(Float.NEGATIVE_INFINITY);
      }

      for (PageRankNode n : values) {
        if (n.getType() == PageRankNode.Type.Structure) {
          // Simply pass along node structure.
          context.write(nid, n);
        } else {
          // Accumulate PageRank mass contributions.
          for (int i = 0; i < sources.length; i++) {
            mass.set(i, sumLogProbs(mass.get(i), n.getPageRank().get(i)));
          }
          massMessages++;
        }
      }

      // Emit aggregated results.
      if (massMessages > 0) {
        intermediateMass.setNodeId(nid.get());
        intermediateMass.setType(PageRankNode.Type.Mass);
        intermediateMass.setPageRank(mass);

        context.write(nid, intermediateMass);
      }
    }
  }

  // Reduce: sums incoming PageRank contributions, rewrite graph structure.
  private static class ReduceClass extends
      Reducer<IntWritable, PageRankNode, IntWritable, PageRankNode> {
    // For keeping track of PageRank mass encountered, so we can compute missing PageRank mass lost
    // through dangling nodes.
    private ArrayListOfFloatsWritable totalMass = new ArrayListOfFloatsWritable();
    private static String[] sources;

    @Override
    public void setup(Context context) throws IOException {
      sources = context.getConfiguration().getStrings(SOURCES);
      for (int i = 0; i < sources.length; i++) {
        totalMass.add(Float.NEGATIVE_INFINITY);
      }
    }

    @Override
    public void reduce(IntWritable nid, Iterable<PageRankNode> iterable, Context context)
        throws IOException, InterruptedException {
      Iterator<PageRankNode> values = iterable.iterator();

      // Create the node structure that we're going to assemble back together from shuffled pieces.
      PageRankNode node = new PageRankNode();

      node.setType(PageRankNode.Type.Complete);
      node.setNodeId(nid.get());

      int massMessagesReceived = 0;
      int structureReceived = 0;

      // Remember, PageRank mass is stored as a log prob.
      ArrayListOfFloatsWritable mass = new ArrayListOfFloatsWritable();

      for (int i = 0; i < sources.length; i++) {
        mass.add(Float.NEGATIVE_INFINITY);
      }

      while (values.hasNext()) {
        PageRankNode n = values.next();

        if (n.getType().equals(PageRankNode.Type.Structure)) {
          // This is the structure; update accordingly.
          ArrayListOfIntsWritable list = n.getAdjacencyList();
          structureReceived++;

          node.setAdjacencyList(list);
        } else {
          // This is a message that contains PageRank mass; accumulate.
          // Accumulate PageRank mass contributions.
          for (int i = 0; i < sources.length; i++) {
            mass.set(i, sumLogProbs(mass.get(i), n.getPageRank().get(i)));
          }
          massMessagesReceived++;
        }
      }

      // Update the final accumulated PageRank mass.
      node.setPageRank(mass);
      context.getCounter(PageRank.massMessagesReceived).increment(massMessagesReceived);

      // Error checking.
      if (structureReceived == 1) {
        // Everything checks out, emit final node structure with updated PageRank value.
        context.write(nid, node);

        // Keep track of total PageRank mass.
        for (int i = 0; i < sources.length; i++) {
          totalMass.set(i, sumLogProbs(totalMass.get(i), mass.get(i)));
        }
      } else if (structureReceived == 0) {
        // We get into this situation if there exists an edge pointing to a node which has no
        // corresponding node structure (i.e., PageRank mass was passed to a non-existent node)...
        // log and count but move on.
        context.getCounter(PageRank.missingStructure).increment(1);
        LOG.warn("No structure received for nodeid: " + nid.get() + " mass: "
            + massMessagesReceived);
        // It's important to note that we don't add the PageRank mass to total... if PageRank mass
        // was sent to a non-existent node, it should simply vanish.
      } else {
        // This shouldn't happen!
        throw new RuntimeException("Multiple structure received for nodeid: " + nid.get()
            + " mass: " + massMessagesReceived + " struct: " + structureReceived);
      }
    }

    @Override
    public void cleanup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      String taskId = conf.get("mapred.task.id");
      String path = conf.get("PageRankMassPath");

      Preconditions.checkNotNull(taskId);
      Preconditions.checkNotNull(path);

      // Write to a file the amount of PageRank mass we've seen in this reducer.
      FileSystem fs = FileSystem.get(context.getConfiguration());
      FSDataOutputStream out = fs.create(new Path(path + "/" + taskId), false);
      for (int i = 0; i < sources.length; i++) {
        out.writeFloat(totalMass.get(i));
      }
      out.close();
    }
  }

  // Mapper that distributes the missing PageRank mass (lost at the dangling nodes) and takes care
  // of the random jump factor.
  private static class MapPageRankMassDistributionClass extends
      Mapper<IntWritable, PageRankNode, IntWritable, PageRankNode> {
    private float[] missing;
    private int nodeCnt = 0;
    private String[] sources;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      sources = conf.getStrings(SOURCES);
      String[] missingMass = conf.getStrings("MissingMass");
      missing = new float[missingMass.length];
      for (int i = 0; i < missing.length; ++i) {
        missing[i] = Float.parseFloat(missingMass[i]);
      }
      nodeCnt = conf.getInt("NodeCount", 0);
    }

    @Override
    public void map(IntWritable nid, PageRankNode node, Context context)
        throws IOException, InterruptedException {
          
      ArrayListOfFloatsWritable p = node.getPageRank();
      for (int i = 0; i < sources.length; i++) {
        int source = Integer.parseInt(sources[i]);
        float rank = p.get(i);
        float jump = Float.NEGATIVE_INFINITY;;
        float link = Float.NEGATIVE_INFINITY;
        if (nid.get() == source) {
          float massMissing = missing[i];
          jump = (float) Math.log(ALPHA);
          link = (float) Math.log(1.0f - ALPHA) + sumLogProbs(rank, (float) (Math.log(massMissing)));
        } else {
          link = rank + (float) (Math.log(1.0f - ALPHA));
        }
        p.set(i,sumLogProbs(jump,link));
      }

      node.setPageRank(p);

      context.write(nid, node);
    }
  }

  // Random jump factor.
  private static float ALPHA = 0.15f;
  private static NumberFormat formatter = new DecimalFormat("0000");

  /**
   * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
   *
   * @param args command-line arguments
   * @throws Exception if tool encounters an exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RunPersonalizedPageRankBasic(), args);
  }

  public RunPersonalizedPageRankBasic() {}

  private static final String BASE = "base";
  private static final String NUM_NODES = "numNodes";
  private static final String START = "start";
  private static final String END = "end";
  private static final String RANGE = "range";
  private static final String SOURCES = "sources";

  /**
   * Runs this tool.
   */
  @SuppressWarnings({ "static-access" })
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(new Option(RANGE, "use range partitioner"));

    options.addOption(OptionBuilder.withArgName("path").hasArg()
        .withDescription("base path").create(BASE));
    options.addOption(OptionBuilder.withArgName("num").hasArg()
        .withDescription("start iteration").create(START));
    options.addOption(OptionBuilder.withArgName("num").hasArg()
        .withDescription("end iteration").create(END));
    options.addOption(OptionBuilder.withArgName("num").hasArg()
        .withDescription("number of nodes").create(NUM_NODES));
    options.addOption(OptionBuilder.withArgName("sources").hasArg()
        .withDescription("number of sources").create(SOURCES));

    CommandLine cmdline;
    CommandLineParser parser = new GnuParser();

    try {
      cmdline = parser.parse(options, args);
    } catch (ParseException exp) {
      System.err.println("Error parsing command line: " + exp.getMessage());
      return -1;
    }

    if (!cmdline.hasOption(BASE) || !cmdline.hasOption(START) ||
        !cmdline.hasOption(END) || !cmdline.hasOption(NUM_NODES) || !cmdline.hasOption(SOURCES)) {
      System.out.println("args: " + Arrays.toString(args));
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(120);
      formatter.printHelp(this.getClass().getName(), options);
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }

    String basePath = cmdline.getOptionValue(BASE);
    int n = Integer.parseInt(cmdline.getOptionValue(NUM_NODES));
    int s = Integer.parseInt(cmdline.getOptionValue(START));
    int e = Integer.parseInt(cmdline.getOptionValue(END));
    boolean useRange = cmdline.hasOption(RANGE);
    String sources = cmdline.getOptionValue(SOURCES);

    LOG.info("Tool name: RunPageRank");
    LOG.info(" - base path: " + basePath);
    LOG.info(" - num nodes: " + n);
    LOG.info(" - start iteration: " + s);
    LOG.info(" - end iteration: " + e);
    LOG.info(" - user range partitioner: " + useRange);
    LOG.info(" - sources: " + sources);

    // Iterate PageRank.
    for (int i = s; i < e; i++) {
      iteratePageRank(i, i + 1, basePath, n, sources);
    }

    return 0;
  }

  // Run each iteration.
  private void iteratePageRank(int i, int j, String basePath, int numNodes,
      String sources) throws Exception {
    // Each iteration consists of two phases (two MapReduce jobs).

    // Job 1: distribute PageRank mass along outgoing edges.
    float[] mass = phase1(i, j, basePath, numNodes, sources);

    // Find out how much PageRank mass got lost at the dangling nodes.
    float[] missing = new float[mass.length];

    for (int k = 0; k < mass.length; k++) {
      LOG.info("mass: " + k + " " + mass[k]);
      missing[k] = 1.0f - (float) (StrictMath.exp(mass[k]));
    }
    // Job 2: distribute missing mass, take care of random jump factor.
    phase2(i, j, missing, basePath, numNodes, sources);
  }

  private float[] phase1(int i, int j, String basePath, int numNodes,
       String sources) throws Exception {
    Job job = Job.getInstance(getConf());
    job.setJobName("PageRank:Basic:iteration" + j + ":Phase1");
    job.setJarByClass(RunPersonalizedPageRankBasic.class);

    String in = basePath + "/iter" + formatter.format(i);
    String out = basePath + "/iter" + formatter.format(j) + "t";
    String outm = out + "-mass";

    // We need to actually count the number of part files to get the number of partitions (because
    // the directory might contain _log).
    int numPartitions = 0;
    for (FileStatus s : FileSystem.get(getConf()).listStatus(new Path(in))) {
      if (s.getPath().getName().contains("part-"))
        numPartitions++;
    }

    LOG.info("PageRank: iteration " + j + ": Phase1");
    LOG.info(" - input: " + in);
    LOG.info(" - output: " + out);
    LOG.info(" - nodeCnt: " + numNodes);
    LOG.info("computed number of partitions: " + numPartitions);

    int numReduceTasks = numPartitions;
    String[] sourceList = sources.split(",");

    job.getConfiguration().setInt("NodeCount", numNodes);
    job.getConfiguration().setBoolean("mapred.map.tasks.speculative.execution", false);
    job.getConfiguration().setBoolean("mapred.reduce.tasks.speculative.execution", false);
    //job.getConfiguration().set("mapred.child.java.opts", "-Xmx2048m");
    job.getConfiguration().set("PageRankMassPath", outm);
    job.getConfiguration().setStrings(SOURCES, sourceList);

    job.setNumReduceTasks(numReduceTasks);

    FileInputFormat.setInputPaths(job, new Path(in));
    FileOutputFormat.setOutputPath(job, new Path(out));

    job.setInputFormatClass(NonSplitableSequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(PageRankNode.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(PageRankNode.class);

    job.setMapperClass(MapClass.class);
    // job.setCombinerClass(CombineClass.class);

    job.setReducerClass(ReduceClass.class);

    FileSystem.get(getConf()).delete(new Path(out), true);
    FileSystem.get(getConf()).delete(new Path(outm), true);

    long startTime = System.currentTimeMillis();
    job.waitForCompletion(true);
    System.out.println("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

    float mass[] = new float[sourceList.length];
    for (int k = 0; k < sourceList.length; k++) {
      mass[k] = Float.NEGATIVE_INFINITY;
    }
    FileSystem fs = FileSystem.get(getConf());
    for (FileStatus f : fs.listStatus(new Path(outm))) {
      FSDataInputStream fin = fs.open(f.getPath());
      for (int k = 0; k < mass.length; k++) {
        mass[k] = sumLogProbs(mass[k], fin.readFloat());
      }
      fin.close();
    }

    return mass;
  }

  private void phase2(int i, int j, float[] missing, String basePath, int numNodes, String sources) throws Exception {
    Job job = Job.getInstance(getConf());
    job.setJobName("PageRank:Basic:iteration" + j + ":Phase2");
    job.setJarByClass(RunPersonalizedPageRankBasic.class);

    LOG.info("number of nodes: " + numNodes);

    String in = basePath + "/iter" + formatter.format(j) + "t";
    String out = basePath + "/iter" + formatter.format(j);

    LOG.info("PageRank: iteration " + j + ": Phase2");
    LOG.info(" - input: " + in);
    LOG.info(" - output: " + out);

    String[] missingMass = new String[missing.length];
    for (int k = 0; k < missing.length; k ++) {
      missingMass[k] = Float.toString(missing[k]);
    }
    String[] sourceList = sources.split(",");
    job.getConfiguration().setBoolean("mapred.map.tasks.speculative.execution", false);
    job.getConfiguration().setBoolean("mapred.reduce.tasks.speculative.execution", false);
    job.getConfiguration().setStrings("MissingMass", missingMass);
    job.getConfiguration().setInt("NodeCount", numNodes);
    job.getConfiguration().setStrings(SOURCES, sourceList);

    job.setNumReduceTasks(0);

    FileInputFormat.setInputPaths(job, new Path(in));
    FileOutputFormat.setOutputPath(job, new Path(out));

    job.setInputFormatClass(NonSplitableSequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(PageRankNode.class);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(PageRankNode.class);

    job.setMapperClass(MapPageRankMassDistributionClass.class);

    FileSystem.get(getConf()).delete(new Path(out), true);

    long startTime = System.currentTimeMillis();
    job.waitForCompletion(true);
    System.out.println("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
  }

  // Adds two log probs.
  private static float sumLogProbs(float a, float b) {
    if (a == Float.NEGATIVE_INFINITY)
      return b;

    if (b == Float.NEGATIVE_INFINITY)
      return a;
    float result;
    if (a < b) {
      result = (float) (b + StrictMath.log1p(StrictMath.exp(a - b)));
    } else {
      result = (float) (a + StrictMath.log1p(StrictMath.exp(b - a)));
    } 
    // if (result > 0.0f) {
    //   return 0.0f;
    // } else {
      return result;
    // }
  }
}