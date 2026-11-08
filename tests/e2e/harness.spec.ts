import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('EnterpriseIQ Browser API Contract Suite (Harness)', () => {
  const harnessUrl = `file://${path.resolve(__dirname, 'harness.html')}`;

  test('Harness Happy Path: User enters question and receives cited answer with PDF source', async ({ page }) => {
    await page.goto(harnessUrl);

    const questionInput = page.getByPlaceholder('Ask a question about enterprise policies...');
    const submitBtn = page.getByRole('button', { name: 'Submit' });

    await expect(questionInput).toBeVisible();
    await questionInput.fill('What is the standard remote work policy for employees?');
    await submitBtn.click();

    // Assert live answer container updates with policy terms
    const answerContainer = page.locator('#answer-text');
    await expect(answerContainer).toContainText(/remote/i, { timeout: 15000 });

    // Assert source citation appears
    const citationContainer = page.locator('#sources-container');
    await expect(citationContainer).toContainText(/\[Source:.*\.pdf\]/i, { timeout: 10000 });
  });

  test('Harness Error State: UI displays error message when backend returns 500 error', async ({ page }) => {
    await page.route('**/ask', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal Server Error' }),
      });
    });

    await page.goto(harnessUrl);

    const questionInput = page.getByPlaceholder('Ask a question about enterprise policies...');
    const submitBtn = page.getByRole('button', { name: 'Submit' });

    await questionInput.fill('Trigger error test');
    await submitBtn.click();

    const answerContainer = page.locator('#answer-text');
    await expect(answerContainer).toHaveClass(/error/, { timeout: 10000 });
    await expect(answerContainer).toContainText(/API Error \(500\)/i);
  });
});

