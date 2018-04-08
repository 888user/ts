package com.tooth.tooth;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Controller;
import org.springframework.stereotype.Service;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

@Service
public class LabelImage {
	
	final static File LOG = new File("log.log");

	public String process(byte[] imageBytes) throws URISyntaxException, IOException {
		FileUtils.writeStringToFile(LOG, "Ini\n", "utf-8", false);

		File file = new ClassPathResource("retrained_graph.pb").getFile();
		
		//InputStream graphStream = LabelImage.class.getResourceAsStream(ruta);
		InputStream graphStream = new FileInputStream(file);
		//InputStream graphStream = LabelImage.class.getResourceAsStream("/retrained_graph.pb");		
		FileUtils.writeStringToFile(LOG, "graphStream=>" + graphStream.toString() + "\n", "utf-8", true);

		byte[] graphDef = IOUtils.toByteArray(graphStream);
		FileUtils.writeStringToFile(LOG, "graphDef\n", "utf-8", true);

		File lableFile = new ClassPathResource("retrained_labels.txt").getFile();
		InputStream lableStream = new FileInputStream(lableFile);
		
		List<String> labels = getLines(lableStream);
		FileUtils.writeStringToFile(LOG, "labels\n", "utf-8", true);

		final File imagenes = new File("Imagenes");
		File resultDir = new File("Resultados");
	
		//FileUtils.writeStringToFile(LOG, "processing..\n", "utf-8", true);
		return processFiles(imageBytes,  graphDef, labels );
//		for (final File image : imagenes.listFiles()) {
//			if (image.isDirectory()) {
//			} else {
//				byte[] imageBytes = readAllBytesOrExit(image.toPath());
//				Tensor<String> image2 = Tensor.create(imageBytes, String.class);
//				File result = new File("Resultados\\Resultado" + ind + ".txt");
//				try (Tensor<Float> tfImage = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
//					float[] labelProbabilities = executeInceptionGraph(graphDef, image2);
//					int bestLabelIdx = maxIndex(labelProbabilities);
//					String res = String.format("%s %s (%.2f%%)", image.getName(), labels.get(bestLabelIdx),
//							labelProbabilities[bestLabelIdx] * 100f);
//					System.out.println(res);
//					try {
//						FileUtils.writeStringToFile(result, res + "\n", "utf-8", true);
//					} catch (IOException e) {
//						e.printStackTrace();
//					}
//				}
//			}
//		}
	//	FileUtils.writeStringToFile(LOG, "End\n", "utf-8", true);

		//showMessage();
	}
	
	public static String processFiles(byte[] imageBytes, byte[] graphDef, List<String> labels ) throws IOException {		
				 
				Tensor<String> image2 = Tensor.create(imageBytes, String.class);
				
				try (Tensor<Float> tfImage = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
					float[] labelProbabilities = executeInceptionGraph(graphDef, image2);
					int bestLabelIdx = maxIndex(labelProbabilities);
					String res = String.format("%s (%.2f%%)", labels.get(bestLabelIdx),
							labelProbabilities[bestLabelIdx] * 100f);
					System.out.println(res);
					try {
						return res;
					} catch (Exception e) {
						e.printStackTrace();
					}
				}	
				return null;

	}

	private static List<String> getLines(InputStream string) throws IOException {
		InputStreamReader reader = new InputStreamReader(string);
		BufferedReader reader2 = new BufferedReader(reader);

		List<String> result = new ArrayList<>();

		String line = null;
		while ((line = reader2.readLine()) != null) {
			result.add(line);
		}
		return result;
	}

	public InputStream getUrl(String name) {
		return getClass().getResourceAsStream(name);
	}

	public static void showMessage() {
		JFrame frame = new JFrame("");
		frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

		JOptionPane.showMessageDialog(frame, "Done!");
		System.exit(0);
	}

	private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
		try (Graph g = new Graph()) {
			GraphBuilder b = new GraphBuilder(g);
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were
			// converted to
			// float using (value - Mean)/Scale.
			final int H = 224;
			final int W = 224;
			final float mean = 117f;
			final float scale = 1f;

			// Since the graph is being constructed once per execution here, we
			// can use a constant for the
			// input image. If the graph were to be re-used for multiple input
			// images, a placeholder would
			// have been more appropriate.
			final Output<String> input = b.constant("input", imageBytes);
			final Output<Float> output = b
					.div(b.sub(
							b.resizeBilinear(b.expandDims(b.cast(b.decodeJpeg(input, 3), Float.class),
									b.constant("make_batch", 0)), b.constant("size", new int[] { H, W })),
							b.constant("mean", mean)), b.constant("scale", scale));
			try (Session s = new Session(g)) {
				return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
			}
		}
	}

	private static float[] executeInceptionGraph(byte[] graphDef, Tensor<String> image) {
		try (Graph g = new Graph()) {
			g.importGraphDef(graphDef);
			try (Session s = new Session(g);

					Tensor<Float> result = s.runner().feed("DecodeJpeg/contents:0", image).fetch("final_result:0").run()
							.get(0).expect(Float.class)) {
				final long[] rshape = result.shape();
				if (result.numDimensions() != 2 || rshape[0] != 1) {
					throw new RuntimeException(String.format(
							"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
							Arrays.toString(rshape)));
				}
				int nlabels = (int) rshape[1];
				return result.copyTo(new float[1][nlabels])[0];
			}
		}
	}

	private static int maxIndex(float[] probabilities) {
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i) {
			if (probabilities[i] > probabilities[best]) {
				best = i;
			}
		}
		return best;
	}

	private static byte[] readAllBytesOrExit(Path path) {
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(1);
		}
		return null;
	}

	private static List<String> readAllLinesOrExit(Path path) {
		try {
			return Files.readAllLines(path, Charset.forName("UTF-8"));
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(0);
		}
		return null;
	}

	// In the fullness of time, equivalents of the methods of this class should
	// be auto-generated from
	// the OpDefs linked into libtensorflow_jni.so. That would match what is
	// done in other languages
	// like Python, C++ and Go.
	static class GraphBuilder {
		GraphBuilder(Graph g) {
			this.g = g;
		}

		Output<Float> div(Output<Float> x, Output<Float> y) {
			return binaryOp("Div", x, y);
		}

		<T> Output<T> sub(Output<T> x, Output<T> y) {
			return binaryOp("Sub", x, y);
		}

		<T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
			return binaryOp3("ResizeBilinear", images, size);
		}

		<T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
			return binaryOp3("ExpandDims", input, dim);
		}

		<T, U> Output<U> cast(Output<T> value, Class<U> type) {
			DataType dtype = DataType.fromClass(type);
			return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().<U> output(0);
		}

		Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
			return g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents).setAttr("channels", channels).build()
					.<UInt8> output(0);
		}

		<T> Output<T> constant(String name, Object value, Class<T> type) {
			try (Tensor<T> t = Tensor.<T> create(value, type)) {
				return g.opBuilder("Const", name).setAttr("dtype", DataType.fromClass(type)).setAttr("value", t).build()
						.<T> output(0);
			}
		}

		Output<String> constant(String name, byte[] value) {
			return this.constant(name, value, String.class);
		}

		Output<Integer> constant(String name, int value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Integer> constant(String name, int[] value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Float> constant(String name, float value) {
			return this.constant(name, value, Float.class);
		}

		private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T> output(0);
		}

		private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T> output(0);
		}

		private Graph g;
	}
}
