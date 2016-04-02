require 'bundler/setup' 
require 'pry'
require 'ruby-fann' 
require 'gnuplot'

inputs = 0.step(Math::PI * 2, 0.01).map { |x| [x] }
desired_outputs = inputs.map { |i| [Math.sin(i.first)] }
train = RubyFann::TrainData.new(inputs: inputs, desired_outputs: desired_outputs)
fann = RubyFann::Standard.new(:num_inputs=>1, :hidden_neurons=>[10, 20], :num_outputs=>1)
fann.set_activation_function_hidden(:sigmoid_symmetric)
fann.set_activation_function_output(:sigmoid_symmetric)
fann.train_on_data(train, 2000, 20, 0.001)

Gnuplot.open do |gnuplot|
  Gnuplot::Plot.new(gnuplot) do |plot|
    plot.terminal 'aqua'
    # plot.output 'fann_sin.png'
    plot.xlabel 'x'
    plot.ylabel 'y'
 
    x = inputs.map(&:first)
    begin
      y = desired_outputs.map(&:first)
  
      plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
        ds.with = "lines"
        ds.notitle
      end
    end
    begin
      y = x.map { |i| fann.run([i]).first }
  
      plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
        ds.with = "lines"
        ds.notitle
      end
    end
  end
end

inputs.map(&:first).map { |i| [i, Math.sin(i), fann.run([i]).first] }.map { |a| a.concat [a[1] - a[2]] }
