import arcjet, { slidingWindow } from "@arcjet/next";

const aj = arcjet({
  key: process.env.ARCJET_KEY,
  characteristics: ["userId"],
  rules: [
    slidingWindow({
      mode: "LIVE",
      interval: 60,
      max: 2,
    }),
  ],
});

export default aj;

// Separate limit for Copilot chat, so chatting doesn't use up the transaction limit
// (and each question costs several Gemini calls on the free tier).
export const copilotAj = arcjet({
  key: process.env.ARCJET_KEY,
  characteristics: ["userId"],
  rules: [
    slidingWindow({
      mode: "LIVE",
      interval: 60,
      max: 6,
    }),
  ],
});
