package digitsresilient;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationBipolarSteepenedSigmoid;
import org.encog.engine.network.activation.ActivationElliottSymmetric;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.BoundMath;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class DigitsResilient {

	private static final long MAX_TRAINING_TIME = 1000;

	private static final int MAX_EPOCHS = 300000;

	private static final double TARGET_ANN_ERROR = 0.0001;

	public static final int TRAINING_SET_SIZE = 1593, INPUT_SIZE = 256, HIDDEN_SIZE = 64, OUTPUT_SIZE = 10;

	private static final NeuralDataSet ZERO_ONE_TRAINING;

	private static final NeuralDataSet MINUS_PLUS_ONE_TRAINING;

	static {
		double[][] trainingZeroOne = new double[TRAINING_SET_SIZE][INPUT_SIZE];
		double[][] idealZeroOne = new double[TRAINING_SET_SIZE][OUTPUT_SIZE];

		double[][] trainingMinusPlusOne = new double[TRAINING_SET_SIZE][INPUT_SIZE];
		double[][] idealMinusPlusOne = new double[TRAINING_SET_SIZE][OUTPUT_SIZE];

		try (FileReader fr = new FileReader("data\\digits.txt"); BufferedReader br = new BufferedReader(fr)) {
			String line;
			for (int i = 0; (line = br.readLine()) != null; ++i) {
				for (int j = 0; j < INPUT_SIZE; ++j) {
					char c = line.charAt(j);
					trainingZeroOne[i][j] = c == '0' ? 0.01 : 0.99;
					trainingMinusPlusOne[i][j] = c == '0' ? -0.99 : 0.99;
				}
				for (int j = 0; j < OUTPUT_SIZE; ++j) {
					char c = line.charAt(INPUT_SIZE + j);
					idealZeroOne[i][j] = c == '0' ? 0.01 : 0.99;
					idealMinusPlusOne[i][j] = c == '0' ? -0.99 : 0.99;
				}

			}
		} catch (IOException ex) {
			Logger.getLogger(DigitsResilient.class.getName()).log(Level.SEVERE, null, ex);
		}

		ZERO_ONE_TRAINING = new BasicNeuralDataSet(trainingZeroOne, idealZeroOne);

		MINUS_PLUS_ONE_TRAINING = new BasicNeuralDataSet(trainingMinusPlusOne, idealMinusPlusOne);
	}

	private static void experiment(String title, ActivationFunction activation, NeuralDataSet training, double epsilon) {
		for (long g = 0; g < 10; g++) {
			BasicNetwork network = new BasicNetwork();
			network.addLayer(new BasicLayer(activation, true, INPUT_SIZE));
			network.addLayer(new BasicLayer(activation, true, HIDDEN_SIZE));
			network.addLayer(new BasicLayer(activation, false, OUTPUT_SIZE));
			network.getStructure().finalizeStructure();
			network.reset();

			final Train train = new ResilientPropagation(network, training);

			int epoch = 1;

			System.out.println(title + "\t" + g);

			for (int i = 0; i < 60; ++i) {
				long start = System.currentTimeMillis();
				do {
					train.iteration();
					epoch++;
				} while (train.getError() > epsilon && (System.currentTimeMillis() - start) < MAX_TRAINING_TIME && epoch < MAX_EPOCHS);

				System.out.println(train.getError());
			}

			System.out.println("epochs:\t" + epoch);
			System.out.println();
			train.finishTraining();
		}
	}

	public static void main(final String args[]) {
		experiment("Fading Sine", new ActivationFadingSin(1), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
		experiment("Exponent Regulated Sin", new ActivationExponentRegulatedSin(1), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
		experiment("Sigmoid", new ActivationSigmoid(), ZERO_ONE_TRAINING, TARGET_ANN_ERROR);
		experiment("Bipolar Sigmoid", new ActivationBipolarSteepenedSigmoid(), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
		experiment("Logarithm", new ActivationLOG(), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
		experiment("Hyperbolic Tangent", new ActivationTANH(), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
		experiment("Elliott Symmetric", new ActivationElliottSymmetric(), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);

		Encog.getInstance().shutdown();
	}

}

class ActivationFadingSin implements ActivationFunction {

	private final ActivationSIN SIN = new ActivationSIN();
	private double period = 1.0D;

	public ActivationFadingSin(double period) {
		this.period = period;
	}

	@Override
	public void activationFunction(double[] values, int start, int size) {
		for (int i = start; i < (start + size) && i < values.length; i++) {
			double x = values[i] / period;

			if (x < -Math.PI || x > Math.PI) {
				values[i] = BoundMath.sin(x) / Math.abs(x);
			} else {
				values[i] = BoundMath.sin(x);
			}
		}
	}

	@Override
	public double derivativeFunction(double before, double after) {
		double x = before / period;

		if (x < -Math.PI || x > Math.PI) {
			return BoundMath.cos(x) / Math.abs(x) - BoundMath.sin(x) / (x * Math.abs(x));
		} else {
			return BoundMath.cos(x);
		}
	}

	@Override
	public ActivationFunction clone() {
		return new ActivationFadingSin(period);
	}

	@Override
	public String getFactoryCode() {
		return null;
	}

	@Override
	public String[] getParamNames() {
		return SIN.getParamNames();
	}

	@Override
	public double[] getParams() {
		return SIN.getParams();
	}

	@Override
	public boolean hasDerivative() {
		return true;
	}

	@Override
	public void setParam(int index, double value) {
		SIN.setParam(index, value);
	}

	@Override
	public String getLabel() {
		return "fading sin";
	}

}

class ActivationExponentRegulatedSin implements ActivationFunction {

	private final ActivationSIN SIN = new ActivationSIN();

	private double period = 1.0D;

	static final double LOW = -0.99;

	static final double HIGH = +0.99;

	public ActivationExponentRegulatedSin(double period) {
		this.period = period;
	}

	@Override
	public void activationFunction(double[] values, int start, int size) {
		for (int i = start; i < (start + size) && i < values.length; i++) {
			double x = values[i] / period;

			values[i] = Math.PI * BoundMath.sin(x) / BoundMath.exp(Math.abs(x));
		}
	}

	@Override
	public double derivativeFunction(double before, double after) {
		double x = before / period;

		if (x == 0) {
			return Double.MAX_VALUE;
		}

		return Math.PI * BoundMath.exp(-Math.abs(x)) * (BoundMath.cos(x) * Math.abs(x) - x * BoundMath.sin(x)) / Math.abs(x);
	}

	@Override
	public ActivationFunction clone() {
		return new ActivationExponentRegulatedSin(period);
	}

	@Override
	public String getFactoryCode() {
		return null;
	}

	@Override
	public String[] getParamNames() {
		return SIN.getParamNames();
	}

	@Override
	public double[] getParams() {
		return SIN.getParams();
	}

	@Override
	public boolean hasDerivative() {
		return true;
	}

	@Override
	public void setParam(int index, double value) {
		SIN.setParam(index, value);
	}

	@Override
	public String getLabel() {
		return "exponent regulated sin";
	}
}
