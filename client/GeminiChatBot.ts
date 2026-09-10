/**
 * ============================================================================
 * GEMINI CHATBOT - CLIENTE DE CONVERSACIÓN NATURAL
 * ============================================================================
 * 
 * Gestiona una conversación multi-turno natural, fluida e inteligente utilizando
 * exclusivamente los endpoints de Gemini expuestos por el backend (/api/gemini).
 * 
 * Características clave:
 * 1. Memoria conversacional multi-turno sin requerir estado en el backend (sliding window).
 * 2. Personalidad de conversación natural (DocuHub AI): empática, clara, colaborativa y fluida.
 * 3. Inyección dinámica de contexto (documentos, categorías, perfil de usuario).
 * 4. Suscripciones reactivas para componentes de UI (React, Vue, etc.).
 * 5. Persistencia opcional en localStorage / sessionStorage.
 * 6. Exportación de transcripciones (Markdown, Texto, JSON).
 */

import { ApiClient, ApiClientError, api as defaultApiInstance } from './ApiClient';
import type { GeminiPromptRequest, GeminiPromptResponse, GeminiStatusResponse } from './ApiClient';

export type ChatRole = 'user' | 'assistant' | 'system';

export type ChatBotStatus = 'idle' | 'sending' | 'thinking' | 'error';

export interface ChatMessageUsage {
  promptTokens?: number;
  candidatesTokens?: number;
  totalTokens?: number;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: number;
  status?: 'sending' | 'sent' | 'error';
  error?: string;
  usage?: ChatMessageUsage;
  metadata?: Record<string, unknown>;
}

export interface GeminiChatBotConfig {
  /** Instancia del ApiClient a utilizar. Si no se suministra, usa la instancia compartida 'api'. */
  client?: ApiClient;
  /** Instrucción de sistema para definir el tono, rol y comportamiento del chatbot. */
  systemInstruction?: string;
  /** Nombre del asistente virtual. Por defecto: 'DocuHub AI'. */
  botName?: string;
  /** Nombre del usuario para personalizar el saludo y diálogo. */
  userName?: string;
  /** Rol o perfil del usuario (ej: 'Administrador', 'Colaborador'). */
  userRole?: string;
  /** Modelo de Gemini a invocar. Por defecto: 'gemini-2.5-flash'. */
  model?: string;
  /** Temperatura / creatividad (0.0 a 2.0). Por defecto: 0.7 para conversación natural. */
  temperature?: number;
  /** Límite de tokens de salida devueltos por Gemini. */
  maxOutputTokens?: number;
  /** Límite de mensajes previos a incluir en la ventana de memoria (sliding window). Por defecto: 16. */
  maxHistoryMessages?: number;
  /** Clave para persistir el historial en localStorage. Por defecto: 'docuhub_gemini_chat_history'. null para desactivar. */
  storageKey?: string | null;
  /** Contexto inicial estático o de dominio (información de repositorios, guías, etc.). */
  initialContext?: string | string[];
  /** Mensaje de bienvenida inicial mostrado al iniciar la conversación. */
  welcomeMessage?: string;
  /** API Key opcional si se desea sobreescribir la del backend. */
  apiKey?: string;
}

export interface SendMessageOptions {
  /** Contexto transitorio para este turno específico (ej: resumen de un documento seleccionado). */
  extraContext?: string | string[];
  /** Sobrescribe la temperatura para este mensaje. */
  temperature?: number;
  /** Sobrescribe el modelo para este mensaje. */
  model?: string;
  /** Si es true, el mensaje no se almacena en el historial. */
  skipHistorySave?: boolean;
}

export type ChatBotListener = (messages: ChatMessage[], status: ChatBotStatus) => void;

/**
 * Instrucción de sistema optimizada para interacción fluida, cálida y natural.
 */
export const DEFAULT_NATURAL_SYSTEM_INSTRUCTION = `
Eres DocuHub AI, un asistente conversacional inteligente, elocuente y cercano integrado en el Repositorio Inteligente DocuHub RVD.
Tu objetivo principal es mantener una conversación totalmente fluida, natural, colaborativa y resolutiva con el usuario.

Principios de conversación natural que debes seguir rigurosamente:
1. Tono humano, cálido y profesional: Habla con naturalidad en español, usando un estilo fresco, cortés y fluido. Evita respuestas robóticas, introducciones acartonadas (como "Como modelo de inteligencia artificial...") o disculpas excesivas.
2. Memoria del diálogo activo: Presta atención activa al hilo de la conversación. Si el usuario te menciona su nombre, gustos, preguntas anteriores o preferencias, recuérdalos y refiérete a ellos de forma intuitiva ("Como comentabas antes...", "Respecto a lo que vimos del balance...").
3. Adaptabilidad de respuesta:
   - Para saludos o preguntas cotidianas: sé directo, simpático y conversacional.
   - Para dudas complejas, análisis de documentos o explicaciones técnicas: estructura la respuesta de forma clara y visualmente agradable utilizando Markdown (títulos concisos, negritas, viñetas ordenadas o fragmentos de código si aplica).
4. Proactividad constructiva: Si una duda del usuario se puede enriquecer con un consejo práctico, un paso siguiente o una pregunta que facilite su trabajo, ofrécelo amablemente al final sin ser invasivo.
5. Dominio de DocuHub: Eres experto en gestión documental, verificación RVD, categorización de archivos, síntesis ejecutiva y extracción de conclusiones.
6. Honestidad: Si falta información sobre algún documento específico, aclara de forma cordial qué datos necesitarías o cómo puede proporcionártelos.
`.trim();

/**
 * Sugerencias de inicio para que el usuario pueda interactuar rápidamente.
 */
export const DEFAULT_SUGGESTIONS = [
  "¿Cómo puedo organizar y buscar mis documentos más eficazmente?",
  "¿En qué consiste la verificación RVD y cómo me beneficia?",
  "Ayúdame a redactar un resumen ejecutivo para un informe",
  "¿Qué tipos de archivos y formatos admite el repositorio?",
  "Hola, ¿en qué me puedes colaborar el día de hoy?"
];

/**
 * Generador simple de identificadores únicos para mensajes.
 */
function createMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

export class GeminiChatBot {
  private client: ApiClient;
  private systemInstruction: string;
  private botName: string;
  private userName: string | undefined;
  private userRole: string | undefined;
  private model: string;
  private temperature: number;
  private maxOutputTokens: number | undefined;
  private maxHistoryMessages: number;
  private storageKey: string | null;
  private dynamicContexts: string[] = [];
  private apiKey: string | undefined;
  private welcomeMessage: string;

  private messages: ChatMessage[] = [];
  private status: ChatBotStatus = 'idle';
  private listeners: Set<ChatBotListener> = new Set();
  private suggestedPrompts: string[] = [...DEFAULT_SUGGESTIONS];

  constructor(config: GeminiChatBotConfig = {}) {
    this.client = config.client || defaultApiInstance;
    this.systemInstruction = config.systemInstruction || DEFAULT_NATURAL_SYSTEM_INSTRUCTION;
    this.botName = config.botName || 'DocuHub AI';
    this.userName = config.userName;
    this.userRole = config.userRole;
    this.model = config.model || 'gemini-2.5-flash';
    this.temperature = typeof config.temperature === 'number' ? config.temperature : 0.7;
    this.maxOutputTokens = config.maxOutputTokens;
    this.maxHistoryMessages = config.maxHistoryMessages || 16;
    this.storageKey = config.storageKey === undefined ? 'docuhub_gemini_chat_history' : config.storageKey;
    this.apiKey = config.apiKey;

    if (config.initialContext) {
      this.setContext(config.initialContext);
    }

    this.welcomeMessage =
      config.welcomeMessage ||
      (this.userName
        ? `¡Hola ${this.userName}! Soy ${this.botName}, tu asistente en el repositorio inteligente. ¿En qué puedo ayudarte o qué tema te gustaría explorar hoy?`
        : `¡Hola! Soy ${this.botName}, tu asistente en el repositorio inteligente. ¿En qué puedo ayudarte o qué tema te gustaría explorar hoy?`);

    this.initializeChat();
  }

  /**
   * Carga el historial desde almacenamiento local si está configurado, o inserta el saludo inicial.
   */
  private initializeChat(): void {
    if (this.storageKey && typeof window !== 'undefined' && window.localStorage) {
      try {
        const stored = window.localStorage.getItem(this.storageKey);
        if (stored) {
          const parsed = JSON.parse(stored);
          if (Array.isArray(parsed) && parsed.length > 0) {
            this.messages = parsed;
            return;
          }
        }
      } catch (err) {
        console.warn('[GeminiChatBot] Error al restaurar historial desde localStorage:', err);
      }
    }

    // Saludo inicial amigable
    this.messages = [
      {
        id: createMessageId(),
        role: 'assistant',
        content: this.welcomeMessage,
        timestamp: Date.now(),
        status: 'sent'
      }
    ];
    this.persistHistory();
  }

  /**
   * Guarda los mensajes en localStorage si está habilitado.
   */
  private persistHistory(): void {
    if (!this.storageKey || typeof window === 'undefined' || !window.localStorage) return;
    try {
      window.localStorage.setItem(this.storageKey, JSON.stringify(this.messages));
    } catch (err) {
      console.warn('[GeminiChatBot] No se pudo persistir el historial en localStorage:', err);
    }
  }

  /**
   * Notifica a todos los suscriptores UI sobre cambios en los mensajes o el estado.
   */
  private notifyListeners(): void {
    const copy = [...this.messages];
    const currentStatus = this.status;
    for (const listener of this.listeners) {
      try {
        listener(copy, currentStatus);
      } catch (e) {
        console.error('[GeminiChatBot] Error en listener de suscripción:', e);
      }
    }
  }

  private setStatus(newStatus: ChatBotStatus): void {
    this.status = newStatus;
    this.notifyListeners();
  }

  /**
   * Suscribe un callback que se ejecuta cada vez que cambia el historial o el estado.
   * Devuelve una función para cancelar la suscripción.
   */
  public subscribe(listener: ChatBotListener): () => void {
    this.listeners.add(listener);
    listener([...this.messages], this.status);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Envía un mensaje en la conversación y gestiona la respuesta de Gemini de forma fluida.
   */
  public async sendMessage(promptText: string, options: SendMessageOptions = {}): Promise<ChatMessage> {
    const cleanPrompt = promptText?.trim();
    if (!cleanPrompt) {
      throw new Error('El mensaje no puede estar vacío.');
    }

    // 1. Crear y registrar mensaje del usuario
    const userMessage: ChatMessage = {
      id: createMessageId(),
      role: 'user',
      content: cleanPrompt,
      timestamp: Date.now(),
      status: 'sent'
    };

    if (!options.skipHistorySave) {
      this.messages.push(userMessage);
      this.persistHistory();
    }

    // 2. Crear mensaje provisional de respuesta para reflejar estado en UI
    const assistantMessageId = createMessageId();
    const assistantPlaceholder: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      status: 'sending'
    };

    if (!options.skipHistorySave) {
      this.messages.push(assistantPlaceholder);
    }
    this.setStatus('thinking');

    try {
      // 3. Ensamblaje inteligente del contexto conversacional multi-turno
      const contextBlocks: string[] = [];

      // A. Perfil del interlocutor
      if (this.userName || this.userRole) {
        const userDetails = [
          this.userName ? `Nombre del usuario: ${this.userName}` : null,
          this.userRole ? `Rol en la plataforma: ${this.userRole}` : null,
          `Nombre de tu persona de asistencia: ${this.botName}`
        ]
          .filter(Boolean)
          .join(' | ');
        contextBlocks.push(`[DATOS DEL INTERLOCUTOR]:\n${userDetails}`);
      }

      // B. Contextos de dominio y dinámicos configurados
      if (this.dynamicContexts.length > 0) {
        contextBlocks.push(`[INFORMACIÓN Y CONTEXTO DEL ESPACIO DE TRABAJO]:\n${this.dynamicContexts.join('\n\n')}`);
      }

      // C. Contexto extra para esta consulta específica
      if (options.extraContext) {
        const extraFormatted = Array.isArray(options.extraContext)
          ? options.extraContext.filter(Boolean).join('\n---\n')
          : String(options.extraContext);
        if (extraFormatted.trim().length > 0) {
          contextBlocks.push(`[CONTEXTO ESPECÍFICO DE ESTA SOLICITUD]:\n${extraFormatted.trim()}`);
        }
      }

      // D. Historial conversacional previo (Sliding Window)
      // Excluimos el mensaje placeholder actual y el último mensaje del usuario para no duplicarlo en el prompt
      const pastMessages = this.messages
        .filter((m) => m.id !== assistantMessageId && m.id !== userMessage.id && m.status !== 'error')
        .slice(-this.maxHistoryMessages);

      if (pastMessages.length > 0) {
        const historyLines = pastMessages.map((m) => {
          const speaker = m.role === 'user' ? (this.userName || 'Usuario') : this.botName;
          return `${speaker}: ${m.content}`;
        });
        contextBlocks.push(`[HISTORIAL RECIENTE DE LA CONVERSACIÓN]:\n${historyLines.join('\n\n')}`);
      }

      // 4. Preparar llamada al endpoint de backend /api/gemini
      const requestPayload: GeminiPromptRequest = {
        prompt: cleanPrompt,
        systemInstruction: this.systemInstruction,
        context: contextBlocks.length > 0 ? contextBlocks.join('\n\n=== SEPARADOR DE CONTEXTO ===\n\n') : undefined,
        model: options.model || this.model,
        temperature: options.temperature !== undefined ? options.temperature : this.temperature,
        maxOutputTokens: this.maxOutputTokens,
        apiKey: this.apiKey
      };

      // 5. Ejecutar petición a través de ApiClient
      const response: GeminiPromptResponse = await this.client.askGemini(requestPayload);

      const responseText = response.data?.response || '';

      // 6. Actualizar mensaje del asistente en el historial
      const targetIndex = this.messages.findIndex((m) => m.id === assistantMessageId);
      const finalAssistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: responseText,
        timestamp: Date.now(),
        status: 'sent',
        usage: response.data?.usage
      };

      if (targetIndex !== -1 && !options.skipHistorySave) {
        this.messages[targetIndex] = finalAssistantMessage;
      } else if (!options.skipHistorySave) {
        this.messages.push(finalAssistantMessage);
      }

      this.persistHistory();
      this.setStatus('idle');
      return finalAssistantMessage;
    } catch (error: any) {
      let friendlyError = 'Ocurrió un error inesperado al comunicarme con Gemini AI.';

      if (error instanceof ApiClientError) {
        if (error.status === 401) {
          friendlyError = 'Tu sesión ha caducado. Por favor, vuelve a iniciar sesión.';
        } else if (error.status === 403) {
          friendlyError = 'Tu usuario no cuenta con permisos suficientes para consultar Gemini AI.';
        } else if (error.status === 502) {
          friendlyError = error.message || 'El servicio de Gemini devolvió un error. Verifica la clave API en el servidor.';
        } else {
          friendlyError = error.message || friendlyError;
        }
      } else if (error?.message) {
        friendlyError = error.message;
      }

      // Actualizar mensaje del asistente a estado de error
      const targetIndex = this.messages.findIndex((m) => m.id === assistantMessageId);
      const errorAssistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: `⚠️ ${friendlyError}`,
        timestamp: Date.now(),
        status: 'error',
        error: friendlyError
      };

      if (targetIndex !== -1 && !options.skipHistorySave) {
        this.messages[targetIndex] = errorAssistantMessage;
      } else if (!options.skipHistorySave) {
        this.messages.push(errorAssistantMessage);
      }

      this.persistHistory();
      this.setStatus('error');
      return errorAssistantMessage;
    }
  }

  /**
   * Reintenta enviar el último mensaje emitido por el usuario si falló la respuesta.
   */
  public async retryLastMessage(): Promise<ChatMessage | null> {
    const lastUserIndex = [...this.messages].reverse().findIndex((m) => m.role === 'user');
    if (lastUserIndex === -1) return null;

    const actualIndex = this.messages.length - 1 - lastUserIndex;
    const userMsg = this.messages[actualIndex];

    // Eliminar mensajes posteriores (asistente o errores)
    this.messages = this.messages.slice(0, actualIndex);
    this.persistHistory();
    this.notifyListeners();

    return this.sendMessage(userMsg.content);
  }

  /**
   * Reinicia la sesión de chat borrando el historial y dejando el mensaje de bienvenida.
   */
  public resetSession(): void {
    this.messages = [
      {
        id: createMessageId(),
        role: 'assistant',
        content: this.welcomeMessage,
        timestamp: Date.now(),
        status: 'sent'
      }
    ];
    this.persistHistory();
    this.setStatus('idle');
  }

  /**
   * Limpia totalmente el historial de mensajes.
   */
  public clearHistory(): void {
    this.messages = [];
    this.persistHistory();
    this.setStatus('idle');
  }

  /**
   * Elimina un mensaje específico por su ID.
   */
  public deleteMessage(id: string): boolean {
    const initialLen = this.messages.length;
    this.messages = this.messages.filter((m) => m.id !== id);
    if (this.messages.length !== initialLen) {
      this.persistHistory();
      this.notifyListeners();
      return true;
    }
    return false;
  }

  /**
   * Devuelve una copia de todos los mensajes registrados.
   */
  public getMessages(): ChatMessage[] {
    return [...this.messages];
  }

  /**
   * Devuelve el estado actual de ejecución del chatbot.
   */
  public getStatus(): ChatBotStatus {
    return this.status;
  }

  /**
   * Consulta el estado de configuración de Gemini en el backend (/api/gemini/status).
   */
  public async getGeminiServiceStatus(): Promise<GeminiStatusResponse> {
    return this.client.getGeminiStatus();
  }

  /**
   * Actualiza el perfil del usuario para personalizar la conversación.
   */
  public setUserProfile(userName?: string, userRole?: string): void {
    this.userName = userName;
    this.userRole = userRole;
  }

  /**
   * Define o reemplaza el contexto de dominio dinámico (ej: metadatos de documentos cargados).
   */
  public setContext(context: string | string[]): void {
    if (Array.isArray(context)) {
      this.dynamicContexts = context.filter(Boolean);
    } else if (context && typeof context === 'string') {
      this.dynamicContexts = [context.trim()];
    } else {
      this.dynamicContexts = [];
    }
  }

  /**
   * Añade un fragmento adicional de contexto al conjunto existente.
   */
  public addContext(snippet: string): void {
    if (snippet && typeof snippet === 'string' && snippet.trim().length > 0) {
      this.dynamicContexts.push(snippet.trim());
    }
  }

  /**
   * Devuelve el array de contextos activos.
   */
  public getContext(): string[] {
    return [...this.dynamicContexts];
  }

  /**
   * Limpia todos los contextos dinámicos registrados.
   */
  public clearContext(): void {
    this.dynamicContexts = [];
  }

  /**
   * Modifica la instrucción de sistema del asistente.
   */
  public setSystemInstruction(instruction: string): void {
    if (instruction && instruction.trim().length > 0) {
      this.systemInstruction = instruction.trim();
    }
  }

  public getSystemInstruction(): string {
    return this.systemInstruction;
  }

  /**
   * Devuelve la lista de preguntas sugeridas para iniciar la conversación.
   */
  public getSuggestedPrompts(): string[] {
    return [...this.suggestedPrompts];
  }

  public setSuggestedPrompts(prompts: string[]): void {
    this.suggestedPrompts = [...prompts];
  }

  /**
   * Exporta la transcripción de la conversación en diversos formatos.
   */
  public exportTranscript(format: 'markdown' | 'text' | 'json' = 'markdown'): string {
    if (format === 'json') {
      return JSON.stringify(this.messages, null, 2);
    }

    if (format === 'text') {
      return this.messages
        .map((m) => {
          const author = m.role === 'user' ? (this.userName || 'Usuario') : this.botName;
          const dateStr = new Date(m.timestamp).toLocaleTimeString();
          return `[${dateStr}] ${author}:\n${m.content}\n`;
        })
        .join('\n');
    }

    // Formato Markdown por defecto
    const header = `# Transcripción de Conversación con ${this.botName}\n*Generada el: ${new Date().toLocaleString()}*\n\n---\n\n`;
    const body = this.messages
      .map((m) => {
        const author = m.role === 'user' ? `🧑 **${this.userName || 'Usuario'}**` : `🤖 **${this.botName}**`;
        const time = new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        return `### ${author} _(${time})_\n\n${m.content}\n`;
      })
      .join('\n---\n\n');

    return `${header}${body}`;
  }

  /**
   * Carga una lista previa de mensajes en la memoria del bot.
   */
  public loadHistory(messages: ChatMessage[]): void {
    if (Array.isArray(messages)) {
      this.messages = [...messages];
      this.persistHistory();
      this.notifyListeners();
    }
  }
}

/**
 * Función fábrica para instanciar fácilmente un bot conversacional de Gemini.
 */
export function createChatBot(config?: GeminiChatBotConfig): GeminiChatBot {
  return new GeminiChatBot(config);
}

/**
 * Instancia compartida global del chatbot.
 */
export const chatBot = new GeminiChatBot();

export default GeminiChatBot;
