import {
  checkBudgetAlerts,
  generateMonthlyReports,
  processRecurringTransaction,
  reindexAllUsers,
  reindexUserKnowledge,
  triggerRecurringTransactions,
} from "../../../lib/inngest/functions";
import { inngest } from "../../../lib/inngest/client";
import { serve } from "inngest/next";

// Create an API that serves zero functions
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    checkBudgetAlerts,
    triggerRecurringTransactions,
    processRecurringTransaction,
    generateMonthlyReports,
    reindexUserKnowledge,
    reindexAllUsers,
  ],
});
