

timeline_svg = d3.select("#timeline")

var margin = {right: 50, left: 50},
    width = +timeline_svg.attr("width") - margin.left - margin.right,
    height = +timeline_svg.attr("height");

var x = d3.scaleLinear()
    .domain([1, 4])
    .range([0, width])
    .clamp(true);

console.log(x.ticks(4))

var slider = timeline_svg.append("g")
    .attr("class", "slider")
    .attr("transform", "translate(" + margin.left + "," + height / 2 + ")");


slider.append("line")
    .attr("class", "track")
    .attr("x1", x.range()[0])
    .attr("x2", x.range()[1])
  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
    .attr("class", "track-inset")
  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
    .attr("class", "track-overlay")
    .call(d3.drag()
        .on("start.interrupt", function() { slider.interrupt(); })
        .on("start drag", function() { select_model(x.invert(d3.event.x)); }));

slider.insert("g", ".track-overlay")
    .attr("class", "ticks")
    .attr("transform", "translate(0," + 18 + ")")
  .selectAll("text")
  .data(x.ticks(4))
  .enter().append("text")
    .attr("x", x)
    .attr("text-anchor", "middle")
    .text(function(d) { return "2020-0" + d; });

var handle = slider.insert("circle", ".track-overlay")
    .attr("class", "handle")
    .attr("r", 9);

slider.transition()
    .duration(9000)
    .tween("select_model", function() {
      var i = d3.interpolate(4, 1);
      return function(t) { select_model(i(t)); };
    });

function select_model(loc) {
  console.log(loc);
  if (loc < 2) {
    var date = "2020-01";
  } else if (loc >= 2 && loc < 3) {
    var date = "2020-02";
  } else if (loc >= 3 && loc < 4) {
    var date = "2020-03";
  } else if (loc == 4) {
    var date = "2020-04";
  }
  console.log(date)
  handle.attr("cx", x(loc));
  timeline_svg.style("background-color", d3.hsl(loc, 0.8, 0.8));
}
