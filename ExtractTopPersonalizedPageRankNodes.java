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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import tl.lin.data.pair.PairOfObjectFloat;
import tl.lin.data.queue.TopScoredObjects;

import java.util.HashMap;
import java.util.Map;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

public class ExtractTopPersonalizedPageRankNodes extends Configured implements Tool {
  private static final Logger LOG = Logger.getLogger(ExtractTopPersonalizedPageRankNodes.class);
  private static Map<String, TopScoredObjects<Integer>> global = new HashMap<>();

  private static class MyMapper extends
      Mapper<IntWritable, PageRankNode, IntWritable, FloatWritable> {
    private TopScoredObjects<Integer> queue;
    IntWritable key = new IntWritable();
    FloatWritable value = new FloatWritable();
    private int source;
    private String sourceName;
    private int k;

    @Override
    public void setup(Context context) throws IOException {
      k = context.getConfiguration().getInt("n", 100);
      queue = new TopScoredObjects<>(k);
      source = context.getConfiguration().getInt("SOURCE", 0);
      sourceName = context.getConfiguration().get("name");
    }

    @Override
    public void map(IntWritable nid, PageRankNode node, Context context) throws IOException,
        InterruptedException {
      queue.add(node.getNodeId(), (float) StrictMath.exp(node.getPageRank().get(source)));
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      IntWritable key = new IntWritable();
      FloatWritable value = new FloatWritable();
      TopScoredObjects<Integer> copyQueue = new TopScoredObjects<>(k);
      for (PairOfObjectFloat<Integer> pair : queue.extractAll()) {
        key.set(pair.getLeftElement());
        value.set(pair.getRightElement());
        copyQueue.add(pair.getLeftElement(), pair.getRightElement());
        LOG.info("Key: " + key.get() + " Value:" + value.get());
        context.write(key, value);
      }
      LOG.info("Source: " + sourceName);
      global.put(sourceName, copyQueue);
    }
  }

  private static class MyReducer extends
      Reducer<IntWritable, FloatWritable, IntWritable, Text> {
    private static TopScoredObjects<Integer> queue;
    private String source;

    @Override
    public void setup(Context context) throws IOException {
      int k = context.getConfiguration().getInt("n", 100);
      source = context.getConfiguration().get("name");
      queue = new TopScoredObjects<Integer>(k);
    }
    
    @Override
    public void reduce(IntWritable nid, Iterable<FloatWritable> iterable, Context context)
    throws IOException {
      Iterator<FloatWritable> iter = iterable.iterator();
      queue.add(nid.get(), iter.next().get());
      
      // Shouldn't happen. Throw an exception.
      if (iter.hasNext()) {
        throw new RuntimeException();
      }
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      IntWritable key = new IntWritable();
      Text value = new Text();
      LOG.info("Source: " + source);
      for (PairOfObjectFloat<Integer> pair : queue.extractAll()) {
        key.set(pair.getLeftElement());
        // We're outputting a string so we can control the formatting.
        value.set(String.format("%.5f", pair.getRightElement()));
        context.write(key, value);
      }
    }
  }

  public ExtractTopPersonalizedPageRankNodes() {
  }

  private static final String INPUT = "input";
  private static final String OUTPUT = "output";
  private static final String TOP = "top";
  private static final String SOURCE = "sources";

  /**
   * Runs this tool.
   */
  @SuppressWarnings({ "static-access" })
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(OptionBuilder.withArgName("path").hasArg()
        .withDescription("input path").create(INPUT));
    options.addOption(OptionBuilder.withArgName("path").hasArg()
        .withDescription("output path").create(OUTPUT));
    options.addOption(OptionBuilder.withArgName("num").hasArg()
        .withDescription("top n").create(TOP));
    options.addOption(OptionBuilder.withArgName("num").hasArg()
        .withDescription("sources").create(SOURCE));

    CommandLine cmdline;
    CommandLineParser parser = new GnuParser();

    try {
      cmdline = parser.parse(options, args);
    } catch (ParseException exp) {
      System.err.println("Error parsing command line: " + exp.getMessage());
      return -1;
    }

    if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT) || !cmdline.hasOption(TOP) || !cmdline.hasOption(SOURCE)) {
      System.out.println("args: " + Arrays.toString(args));
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(120);
      formatter.printHelp(this.getClass().getName(), options);
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }

    String inputPath = cmdline.getOptionValue(INPUT);
    String outputPath = cmdline.getOptionValue(OUTPUT);
    int n = Integer.parseInt(cmdline.getOptionValue(TOP));
    String sourceList = cmdline.getOptionValue(SOURCE);

    LOG.info("Tool name: " + ExtractTopPersonalizedPageRankNodes.class.getSimpleName());
    LOG.info(" - input: " + inputPath);
    LOG.info(" - output: " + outputPath);
    LOG.info(" - top: " + n);
    LOG.info(" - sources: " + sourceList);

    String[] sources = sourceList.split(",");
    
    for (int i = 0; i < sources.length; i++) {
      Configuration conf = getConf();
      conf.setInt("mapred.min.split.size", 1024 * 1024 * 1024);
      conf.setInt("n", n);
      conf.setInt("SOURCE", i);
      conf.set("name", sources[i]);

      Job job = Job.getInstance(conf);
      job.setJobName(ExtractTopPersonalizedPageRankNodes.class.getName() + ":" + inputPath);
      job.setJarByClass(ExtractTopPersonalizedPageRankNodes.class);

      job.setNumReduceTasks(1);

      FileInputFormat.addInputPath(job, new Path(inputPath));
      FileOutputFormat.setOutputPath(job, new Path(outputPath + i));

      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputFormatClass(TextOutputFormat.class);

      job.setMapOutputKeyClass(IntWritable.class);
      job.setMapOutputValueClass(FloatWritable.class);

      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(Text.class);
      // Text instead of FloatWritable so we can control formatting

      job.setMapperClass(MyMapper.class);
      // job.setReducerClass(MyReducer.class);

      // Delete the output directory if it exists already.
      FileSystem.get(conf).delete(new Path(outputPath + i), true);

      job.waitForCompletion(true);
    }
    // System.out.println("HERE");
    for (String key : global.keySet()) {
      TopScoredObjects<Integer> queue = global.get(key);
      System.out.println("Source: " + key);
      for (PairOfObjectFloat<Integer> pair : queue.extractAll()) {
        int neighbour = pair.getLeftElement();
        System.out.println(String.format("%.5f " + neighbour, pair.getRightElement()));
      }
      System.out.println("");
    }

    return 0;
  }

  /**
   * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
   *
   * @param args command-line arguments
   * @throws Exception if tool encounters an exception
   */
  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new ExtractTopPersonalizedPageRankNodes(), args);
    System.exit(res);
  }
}




