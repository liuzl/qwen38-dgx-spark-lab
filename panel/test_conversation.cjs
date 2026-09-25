const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
// Run the real formatter functions without starting the DOM polling loop.
const source = fs.readFileSync(__dirname + "/static/app.js", "utf8");
const formatters = source.slice(
  source.indexOf("function contentText("),
  source.indexOf("function messageCard("),
);
const context = vm.createContext({});
vm.runInContext(formatters, context);
const parse = (value) => context.parseResponse(value);
test("chat SSE merges text and tool argument fragments", () => {
  const events =
    [
      {
        choices: [
          {
            index: 0,
            delta: {
              content: "Hello ",
              tool_calls: [
                { index: 0, function: { name: "search", arguments: '{"q":' } },
              ],
            },
          },
        ],
      },
      {
        choices: [
          {
            index: 0,
            delta: {
              content: "world",
              tool_calls: [{ index: 0, function: { arguments: '"test"}' } }],
            },
          },
        ],
      },
    ]
      .map((e) => "data: " + JSON.stringify(e))
      .join("\n\n") + "\n\ndata: [DONE]\n";
  const result = parse(events);
  assert.equal(result.answer, "Hello world");
  assert.equal(result.tools[0].name, "search");
  assert.equal(result.tools[0].arguments, '{"q":"test"}');
});
test("Responses terminal object does not duplicate streamed answer", () => {
  const events = [
    { type: "response.output_text.delta", delta: "OK" },
    {
      type: "response.completed",
      response: {
        output: [
          { type: "message", content: [{ type: "output_text", text: "OK" }] },
        ],
      },
    },
  ];
  assert.equal(
    parse(events.map((e) => "data: " + JSON.stringify(e)).join("\n")).answer,
    "OK",
  );
});
test("JSON reasoning and raw text are preserved without markup execution", () => {
  const result = parse(
    JSON.stringify({
      choices: [
        {
          message: {
            content: "<script>untrusted</script>",
            reasoning_content: "reason",
          },
        },
      ],
    }),
  );
  assert.equal(result.answer, "<script>untrusted</script>");
  assert.equal(result.reasoning, "reason");
  assert.equal(
    context.contentText([
      { type: "image_url", image_url: { url: "data:image/png;base64,secret" } },
      { type: "text", text: "hello" },
    ]),
    "[图片输入]\nhello",
  );
});
test("truncated SSE retains complete previous events", () => {
  assert.equal(
    parse(
      'data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: {"broken',
    ).answer,
    "partial",
  );
});
