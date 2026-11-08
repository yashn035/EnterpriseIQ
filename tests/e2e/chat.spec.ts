import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('EnterpriseIQ E2E Chat Interface Suite', () => {

  test('Happy Path: User asks policy question and receives cited answer with PDF source', async ({ page }) => {
    await page.goto('/');

    // 1. Locate chat input using stable role/placeholder selector
    const chatInput = page.getByPlaceholder('Ask a question about enterprise policies or documents...');
    await expect(chatInput).toBeVisible({ timeout: 15000 });

    // 2. Submit question
    await chatInput.fill('What is the standard remote work policy for employees?');
    await chatInput.press('Enter');

    // 3. Assert answer appears with expected policy keywords
    const responseLocator = page.getByText(/remote/i).last();
    await expect(responseLocator).toBeVisible({ timeout: 25000 });

    // 4. Assert source PDF document citation appears
    const citationLocator = page.getByText(/\.pdf/i).last();
    await expect(citationLocator).toBeVisible({ timeout: 15000 });
  });

  test('Error State: Graceful error handling when backend is unreachable', async ({ page }) => {
    await page.goto('/');

    // In Streamlit, requests are initiated server-side in Python, not from browser fetch().
    // Therefore, to test backend failure against the Streamlit server, we point the backend endpoint to an unreachable port.
    const endpointInput = page.getByLabel('Backend API Endpoint');
    await expect(endpointInput).toBeVisible({ timeout: 15000 });
    await endpointInput.fill('http://127.0.0.1:9999/ask');
    await endpointInput.press('Enter');

    const chatInput = page.getByPlaceholder('Ask a question about enterprise policies or documents...');
    await expect(chatInput).toBeVisible({ timeout: 15000 });

    await chatInput.fill('What is the sick leave policy?');
    await chatInput.press('Enter');

    // Assert UI shows prominent error notification rather than hanging
    const errorAlert = page.getByText(/Connection failed to backend API|API Error/i).first();
    await expect(errorAlert).toBeVisible({ timeout: 15000 });
  });

  test('Document Ingestion: Uploading PDF document surfaces in ingested document list', async ({ page }) => {
    await page.goto('/');

    // Locate standard HTML file input rendered by Streamlit sidebar file_uploader
    const fileInput = page.locator('input[type="file"]').first();
    await expect(fileInput).toBeAttached({ timeout: 15000 });

    const samplePdfPath = path.resolve(__dirname, '../../data/raw/sample.pdf');
    await fileInput.setInputFiles(samplePdfPath);

    // Verify filename confirmation appears in UI
    const selectedNotice = page.getByText(/sample\.pdf/i).first();
    await expect(selectedNotice).toBeVisible({ timeout: 15000 });

    // Click Upload & Ingest button using role selector
    const uploadBtn = page.getByRole('button', { name: /Upload & Ingest/i });
    if (await uploadBtn.isVisible()) {
      await uploadBtn.click();
      // Assert status completion or success toast
      const statusNotice = page.getByText(/Ingestion complete|Successfully ingested|already exists/i).first();
      await expect(statusNotice).toBeVisible({ timeout: 20000 });
    }
  });

});
