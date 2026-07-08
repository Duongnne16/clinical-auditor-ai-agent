import { useState } from 'react'
import type { FormEvent } from 'react'
import { login, register } from '../api/auth'

type AuthMode = 'login' | 'register'

type AuthScreenProps = {
  onAuthenticated: () => void | Promise<void>
}

export default function AuthScreen({ onAuthenticated }: AuthScreenProps) {
  const [mode, setMode] = useState<AuthMode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  const isRegisterMode = mode === 'register'

  const handleModeChange = (nextMode: AuthMode) => {
    setMode(nextMode)
    setErrorMessage('')
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (isSubmitting) {
      return
    }

    setIsSubmitting(true)
    setErrorMessage('')

    try {
      if (isRegisterMode) {
        await register({
          email,
          password,
          full_name: fullName.trim() || null,
        })
      } else {
        await login({ email, password })
      }

      await onAuthenticated()
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Không thể xác thực tài khoản. Vui lòng thử lại.',
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-10 text-gray-950">
      <section className="w-full max-w-md rounded-3xl border border-gray-200 bg-white px-6 py-8 shadow-[0_18px_60px_rgba(15,23,42,0.08)] sm:px-8">
        <div className="text-center">
          <h1 className="text-3xl font-semibold tracking-normal text-gray-950">
            Hệ thống kiểm tra đơn thuốc
          </h1>
          <p className="mt-3 text-sm leading-6 text-gray-500">
            Đăng nhập hoặc tạo tài khoản để sử dụng hệ thống kiểm tra đơn thuốc.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-2 rounded-full bg-gray-100 p-1">
          <button
            type="button"
            onClick={() => handleModeChange('login')}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-gray-300 ${
              !isRegisterMode
                ? 'bg-white text-gray-950 shadow-sm'
                : 'text-gray-500 hover:text-gray-900'
            }`}
          >
            Đăng nhập
          </button>
          <button
            type="button"
            onClick={() => handleModeChange('register')}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-gray-300 ${
              isRegisterMode
                ? 'bg-white text-gray-950 shadow-sm'
                : 'text-gray-500 hover:text-gray-900'
            }`}
          >
            Đăng ký
          </button>
        </div>

        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
          {isRegisterMode ? (
            <label className="block">
              <span className="text-sm font-medium text-gray-700">
                Họ và tên
              </span>
              <input
                type="text"
                value={fullName}
                onChange={(event) => setFullName(event.target.value)}
                disabled={isSubmitting}
                autoComplete="name"
                className="mt-2 w-full rounded-2xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-950 outline-none transition placeholder:text-gray-400 focus:border-gray-400 focus:ring-4 focus:ring-gray-100 disabled:cursor-not-allowed disabled:bg-gray-50"
                placeholder="Nguyễn Văn A"
              />
            </label>
          ) : null}

          <label className="block">
            <span className="text-sm font-medium text-gray-700">Email</span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              disabled={isSubmitting}
              required
              autoComplete="email"
              className="mt-2 w-full rounded-2xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-950 outline-none transition placeholder:text-gray-400 focus:border-gray-400 focus:ring-4 focus:ring-gray-100 disabled:cursor-not-allowed disabled:bg-gray-50"
              placeholder="doctor@example.com"
            />
          </label>

          <label className="block">
            <span className="text-sm font-medium text-gray-700">Mật khẩu</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              disabled={isSubmitting}
              required
              minLength={isRegisterMode ? 8 : 1}
              autoComplete={isRegisterMode ? 'new-password' : 'current-password'}
              className="mt-2 w-full rounded-2xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-950 outline-none transition placeholder:text-gray-400 focus:border-gray-400 focus:ring-4 focus:ring-gray-100 disabled:cursor-not-allowed disabled:bg-gray-50"
              placeholder={isRegisterMode ? 'Ít nhất 8 ký tự' : 'Mật khẩu'}
            />
          </label>

          {errorMessage ? (
            <p className="rounded-2xl border border-red-100 bg-red-50 px-4 py-3 text-sm leading-5 text-red-700">
              {errorMessage}
            </p>
          ) : null}

          <button
            type="submit"
            disabled={isSubmitting}
            className="mt-2 flex w-full items-center justify-center rounded-2xl bg-black px-4 py-3 text-sm font-semibold text-white transition hover:bg-gray-800 focus:outline-none focus:ring-4 focus:ring-gray-200 disabled:cursor-not-allowed disabled:bg-gray-400"
          >
            {isSubmitting
              ? 'Đang xử lý...'
              : isRegisterMode
                ? 'Tạo tài khoản'
                : 'Đăng nhập'}
          </button>
        </form>
      </section>
    </main>
  )
}
