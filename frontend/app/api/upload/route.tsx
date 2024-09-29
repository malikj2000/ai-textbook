import { NextRequest, NextResponse } from 'next/server';
import formidable from 'formidable';
import fs from 'fs';
import pdfParse from 'pdf-parse';
import { IncomingMessage } from 'http';

export const config = {
    api: {
        bodyParser: false,
    }
}

export async function POST(request: NextRequest) {
    const nodeRequest = request as unknown as IncomingMessage;
    return new Promise((resolve, reject) => {
        const form = new formidable.IncomingForm();

        form.parse(nodeRequest, async (err, fields, files) => {
            if (err) {
                return reject(err);
            }

            const file = files.file?.[0];
            if (!file) {
                throw new Error('No file uploaded');
            }
            const filePath = file.filepath;

            const dataBuffer = fs.readFileSync(filePath);
            const extractedText = await pdfParse(dataBuffer);

            await processAndStoreText(extractedText.text);

            resolve(NextResponse.json({ message: 'File uploaded and processed successfully' }));
        });
    });
}

async function processAndStoreText(text: string): Promise<void> {
    const chunks = splitTextIntoChunks(text);
    for (const chunk of chunks) {
        const embedding = await getEmbeddingForChunk(chunk);
        await storeEmbeddingInVectorDatabase(chunk, embedding);
    }
}

function splitTextIntoChunks(text: string, chunkSize=500): Array<string> {
    const sentences = text.split('. ');
    const chunks = [];
    let chunk = '';

    for (const sentence of sentences) {
        if (chunk.length + sentence.length < chunkSize) {
            chunk += sentence + '. ';
        } else {
            chunks.push(chunk);
            chunk = sentence + '. ';
        }
    }

    if (chunk) chunks.push(chunk);
    return chunks;
}
